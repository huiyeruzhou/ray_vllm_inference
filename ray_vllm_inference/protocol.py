from typing import List, Optional, Literal
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
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    ignore_eos: Optional[bool] = False


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
