from typing import List, Optional, Literal, Union, Dict
from pydantic import BaseModel, model_validator


class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content
    
class GenerateRequest(BaseModel):
    """Generate completion request.

    prompt: Prompt to use for the generation
    messages: List of messages to use for the generation
    stream: Bool flag whether to stream the output or not
    max_tokens: Maximum number of tokens to generate per output sequence.
    temperature: Float that controls the randomness of the sampling. Lower
        values make the model more deterministic, while higher values make
        the model more random. Zero means greedy sampling.
    ignore_eos: Whether to ignore the EOS token and continue generating
        tokens after the EOS token is generated.

    Note that vLLM supports many more sampling parameters that are ignored here.
    See: vllm/sampling_params.py in the vLLM repository.
    """
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    best_of: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[int] = None
    # bad_words: Optional[List[str]] = None # will be implemented in the future version of vllm, now is 0.5.4
    # logit_bias: Optional[Union[Dict[int, float], Dict[str, float]]] = None
    # allowed_token_ids: Optional[List[int]] = None


    @model_validator(mode="after")
    def check_prompt_messages(cls, values):
        """Check that either prompt or messages is set, but not both."""
        if values.prompt is not None and values.messages is not None:
            raise ValueError(f"Either prompt or messages must be set, not both: {values.prompt=}, {values.messages=}")
        if values.prompt is None and values.messages is None:
            raise ValueError(f"Neither prompt nor messages is set, one of them must be set {values=}")
        return values

class GenerateResponse(BaseModel):
    texts: list[str]
    prompt_tokens: int
    output_tokens: list[int]
    finish_reason: Optional[list[Optional[str]]] = None
