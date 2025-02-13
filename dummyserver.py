# 起一个fastapi server，监听1080/generate，返回所有请求参数

import json
import fastapi
import uvicorn
import logging
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse

class Request(BaseModel):
    prompt: str = None
    model: str = None
    temperature: float = None
    max_tokens: int = None
    top_p: float = None
    frequency_penalty: float = None
    presence_penalty: float = None
    stop: Optional[list[str]] = None
    stream: bool = None

app = FastAPI()

@app.post("/generate")
def generate(request: Request):
    logging.info(f"request: {request}")
    return JSONResponse(content=request.dict())

if __name__ == "__main__":
    uvicorn.run(app, port=8000)