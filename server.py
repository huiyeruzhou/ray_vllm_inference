import ray
import ray.serve as serve
from ray_vllm_inference.vllm_serve import VLLMGenerateDeployment
import hydra
from omegaconf import OmegaConf

def get_available_gpus():
    return ray.available_resources().get("GPU", 0)

@hydra.main(config_path="conf", config_name="infer_server", version_base=None)
def main(cfg):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    import ray
    ray.init(
        runtime_env={
            "pip": ["fastapi -U"],
            'working_dir': '.',
            "excludes": [
                "outputs",
            ],
        }
    )
    if cfg.num_workers * cfg.vllm.tensor_parallel_size > get_available_gpus():
        raise ValueError(f"Number of GPUs in config ({cfg.num_workers * cfg.vllm.tensor_parallel_size=}) > available gpus in ray ({get_available_gpus()})")
    
    # import the deployment class
    import importlib
    module_path, class_name = cfg.deployment_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    DeploymentClass = getattr(module, class_name)
    if not isinstance(DeploymentClass, serve.Deployment):
        raise ValueError(f"Deployment class {cfg.deployment_class} is not a subclass of serve.Deployment, maybe forget to decorate it with @serve.deployment ?")
    else:
        print(f"Deployment: {cfg.deployment_class}")

    app : serve.Application = DeploymentClass.options(num_replicas=cfg.num_workers, ray_actor_options={"num_gpus": cfg.vllm.tensor_parallel_size}, **cfg.deployment).bind(cfg)
    import signal
    def sigint_handler(signum, frame):
        print("User interrupted the program.")
        serve.delete(cfg.name, False)
        print("Deployment will be deleted asynchronously.")
        exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    serve.run(app, blocking=True,  name=cfg.name)


if __name__ == "__main__":
    main()