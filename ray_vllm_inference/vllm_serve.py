from typing import Dict, List, AsyncGenerator
import logging
import uuid
from http import HTTPStatus
from ray import serve
from ray.serve import Application
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response, JSONResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from ray_vllm_inference.protocol import GenerateRequest, GenerateResponse
from omegaconf import DictConfig
logger = logging.getLogger("ray.serve")

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return create_error_response(HTTPStatus.BAD_REQUEST, f'Error parsing JSON payload: {exc}')

def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code.value, content={"detail":message})

@serve.deployment(name='VLLMInference', 
                  num_replicas=1, 
                  max_ongoing_requests=256,
                  ray_actor_options={"num_gpus": 1.0})
@serve.ingress(app)
class VLLMGenerateDeployment:
    def __init__(self, config: DictConfig):
        """
        Construct a VLLM deployment.

        Args:
            model: name or path of the huggingface model to use
            download_dir: directory to download and load the weights,
                default to the default cache dir of huggingface.
            load_format: The format of the model weights to load.
                "auto" will try to load the weights in the safetensors format and fall 
                back to the pytorch bin format if safetensors format is not available.
                "pt" will load the weights in the pytorch bin format.
                "safetensors" will load the weights in the safetensors format.
                "npcache" will load the weights in pytorch format and store a numpy 
                cache to speed up the loading.
                "dummy" will initialize the weights with random values, which is mainly 
                for profiling.
            dtype: data type for model weights and activations.
                The "auto" option will use FP16 precision
                for FP32 and FP16 models, and BF16 precision.
                for BF16 models.
            max_model_len: model context length. If unspecified, will be automatically 
                derived from the model.
            worker_use_ray: use Ray for distributed serving, will be
                automatically set when using more than 1 GPU
            pipeline_parallel_size: number of pipeline stages.
            tensor_parallel_size: number of tensor parallel replicas.
            block_size: token block size.
            swap_space: CPU swap space size (GiB) per GPU.
            gpu_memory_utilization: the percentage of GPU memory to be used for
                the model executor
            max_num_batched_tokens: maximum number of batched tokens per iteration
            max_num_seqs: maximum number of sequences per iteration.
            disable_log_stats: disable logging statistics.
            quantization: method used to quantize the weights
            engine_use_ray: use Ray to start the LLM engine in a separate
                process as the server process.
            disable_log_requests: disable logging requests.
        """
        args = AsyncEngineArgs(**config.vllm)
        logger.info(args)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        engine_model_config = self.engine.engine.get_model_config()
        self.tokenizer = self.engine.engine.get_tokenizer()
        self.max_model_len = config.vllm.get('max_model_len', engine_model_config.max_model_len)


    def _next_request_id(self):
        return str(uuid.uuid1().hex)

    def _check_length(self, prompt:str, request:GenerateRequest) -> List[int]:
        input_ids = self.tokenizer(prompt).input_ids
        token_num = len(input_ids)

        if request.max_tokens is None:
            request.max_tokens = self.max_model_len - token_num
        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
            f"This model's maximum context length is {self.max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.")
        return input_ids

    async def _stream_results(self, output_generator) -> AsyncGenerator[bytes, None]:
        num_returned_texts = []
        num_returned_tokens = []
        async for request_output in output_generator:
            outputs = request_output.outputs
            if not num_returned_texts:
                num_returned_texts = [0] * len(outputs)
                num_returned_tokens = [0] * len(outputs)
            response = GenerateResponse(texts=[output.text[num_returned_texts[i]:] for i, output in enumerate(outputs)],
                                        prompt_tokens=len(request_output.prompt_token_ids),
                                        output_tokens=[len(output.token_ids) - num_returned_tokens[i] for i, output in enumerate(outputs)],
                                        finish_reason=[output.finish_reason for output in outputs])
            yield (response.model_dump_json() + "\n").encode("utf-8")
            for i, output in enumerate(outputs):
                num_returned_texts[i] = len(output.text)
                num_returned_tokens[i] = len(output.token_ids)

    async def _abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    @app.post("/generate")
    async def generate(self, request:GenerateRequest, raw_request:Request) -> Response:
        """Generate completion for the request.

        Args:
            request (GenerateRequest): Request object
            raw_request (Request): FastAPI request object

        Returns:
            Response: Response object
        """
        try:
            if not request.prompt and not request.messages:
                return create_error_response(HTTPStatus.BAD_REQUEST, "Missing parameter 'prompt' or 'messages'")

            if request.prompt:
                prompt = request.prompt
            else:
                prompt = self.tokenizer.apply_chat_template(request.messages,
                                                            add_generation_prompt=True,
                                                            tokenize=False)

            # prompt_token_ids = self._check_length(prompt, request)

            request_dict = request.model_dump(exclude=set(['prompt', 'messages', 'stream']))

            sampling_params = SamplingParams(**request_dict)
            def parse_stop(self, stop):
                if isinstance(stop, str):
                    stop = [stop]
                if "<|EOS_TOKEN|>" in stop:
                    eos_id = self.tokenizer.eos_token_id
                    stop.remove("<|EOS_TOKEN|>")
                    if isinstance(eos_id, int):
                        stop.append(self.tokenizer.decode(eos_id))
                    elif isinstance(eos_id, list):
                        stop.extend([self.tokenizer.decode(e) for e in eos_id])
                    else:
                        raise ValueError(f"Invalid eos_id type: {type(eos_id)}")
                return stop
            if sampling_params.stop:
                sampling_params.stop = parse_stop(self, sampling_params.stop)
            
            request_id = self._next_request_id()
            
            ## TODO: if we are using higher version of vllm, set output_kind for a better output_generator
            # DELTA for stream, FINAL for non-stream
            output_generator = self.engine.generate(inputs=prompt,
                                    sampling_params=sampling_params, 
                                    request_id=request_id, 
                                    )
            if request.stream:
                background_tasks = BackgroundTasks()
                # Abort the request processing in the engine if the socket connection drops
                background_tasks.add_task(self._abort_request, request_id)
                return StreamingResponse(self._stream_results(output_generator), 
                                        background=background_tasks)

            else:
                final_output = None
                async for request_output in output_generator:
                    if await raw_request.is_disconnected():
                        await self.engine.abort(request_id)
                        return Response(status_code=200)
                    final_output = request_output

                texts = [out.text for out in final_output.outputs]
                prompt_tokens = len(final_output.prompt_token_ids)
                output_tokens = [len(out.token_ids) for out in final_output.outputs]
                finish_reason = [out.finish_reason for out in final_output.outputs]
                return GenerateResponse(texts=texts, prompt_tokens=prompt_tokens, 
                                        output_tokens=output_tokens, finish_reason=finish_reason)

        except ValueError as e:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            logger.error('Error in generate()', exc_info=1)
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, 'Server error')

def deployment(args: Dict[str, str]) -> Application:
    yaml = args.get('config', 'conf/infer_server.yaml')
    import omegaconf
    config = omegaconf.OmegaConf.load(yaml)
    return VLLMGenerateDeployment.bind(config)

APP = deployment({})