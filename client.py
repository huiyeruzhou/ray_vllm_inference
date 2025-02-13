# client.py
import pandas as pd
import asyncio
import uvloop
import json
import time
from typing import  Dict
from collections import Counter
import traceback
from omegaconf import DictConfig, OmegaConf
import hydra
from ray_vllm_inference.protocol import GenerateResponse
from typing import Awaitable

# use event loop of uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import logging

logger = logging.getLogger(__name__)

config = None

from api import APIClient
from methods.tree_search import tree_search
from methods.cot import cot



async def client_request(
    api_client: APIClient,
    async_method: Awaitable,
    async_method_args: DictConfig,
    output_dir: str,
    client_id: int,
    data: dict,
):
    try:
        start_time = time.time()
        token_counter = await async_method(api_client, client_id, output_dir, data, **async_method_args)
        end_time = time.time()
        time_elapsed = end_time - start_time
        logger.info(
            f"""Client {client_id} received result (Time taken: {time_elapsed:.2f}s). Input tokens: {token_counter.get('input_tokens',0)}, Output tokens: {token_counter.get('output_tokens',0)}."""
        )
        return token_counter
    except asyncio.CancelledError:
        raise
    except TimeoutError as e:
        logger.error(f"Client {client_id} timed out: {e}")
    except Exception as e:
        logger.error(f"Client {client_id} encountered an error: {e}")
        raise


def load_dataset(config):
    with open(config.dataset.path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()][: config.dataset.max_length]

    return pd.DataFrame(
        [
            {
                "prompt": """Solve the following math problem step by step. Use double newlines as the end of each step. The last line of your response should be of the form \\boxed{{ANSWER}} where ANSWER is the answer to the problem.

{Question}

Remember to put your answer within \\boxed{{}}.
""".format(
                    Question=d["problem"]
                ),
                **d,
            }
            for d in data
        ]
    )


async def main_async(df: pd.DataFrame):
    api_client = APIClient(config=config.http)
    try:
        # make sure the server is connected
        await api_client.test()
    except Exception as e:
        logger.error(f"Test request failed: {e}")
        raise

    async_method = globals()[config.method.name]
    if not asyncio.iscoroutinefunction(async_method):
        raise ValueError(f"{config.method} is not a coroutine function, please check config.method, or use `async def`")
    async_method_args = config.method.args
    try:
        async with asyncio.TaskGroup() as tg:
            for i, data in enumerate(df.to_dict("records")):
                tg.create_task(client_request(api_client, async_method, async_method_args, config.dataset.output_dir, i, data))
        api_client.perf(task=len(df.to_dict("records")))
    except asyncio.CancelledError:
        logger.error("Main task was cancelled. Cancelling all client tasks...")
        raise
    except Exception as e:
        logger.error(f"Main task exception: {e}")
        traceback.print_exc()
        exit(1)
    finally:
        await api_client.close()


def evaluate(args):
    from verifiers.math_verifier import is_correct_minerva

    i, ans = args
    global config
    import os

    with open(os.path.join(config.dataset.output_dir, f"{i}.json"), "r") as f:
        data = json.load(f)

    result = data["result"]
    score, pred = is_correct_minerva(result, ans)
    return {"id": i, "result": result, "score": score, "pred": pred}


if __name__ == "__main__":
    import os

    @hydra.main(config_path="conf", config_name="client", version_base=None)
    def main(cfg: DictConfig):
        global config
        config = cfg

        OmegaConf.resolve(config)
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))

        df = load_dataset(config)

        os.makedirs(config.dataset.output_dir, exist_ok=True)
        logger.info(f"Output directory: {config.dataset.output_dir}")

        try:
            asyncio.run(main_async(df))
            from tqdm.contrib.concurrent import process_map

            scores = process_map(evaluate, enumerate(df["answer"]), max_workers=32)
            logger.info(f"Average score: {sum(s['score'] for s in scores) / len(scores)}")

            with open(os.path.join(config.dataset.output_dir, "scores.json"), "w") as f:
                json.dump(scores, f, indent=4)
        except KeyboardInterrupt:
            logger.info("Interrputed by user (Ctrl+C)")

    main()
