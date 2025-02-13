# client.py
import asyncio
import uvloop
import time
from typing import List, Dict
from collections import Counter, defaultdict
import httpx
from omegaconf import OmegaConf
from ray_vllm_inference.protocol import GenerateResponse

# use event loop of uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import logging

logger = logging.getLogger(__name__)

config = None

class APIClient:
    """
    A simple API client for different types of generate requests.
    """

    def __init__(self, config):
        self.server_url = config.url
        if self.server_url.endswith("/"):
            self.server_url = self.server_url[:-1]
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(5, read=config.read_timeout),
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections,
                keepalive_expiry=config.keepalive_expiry,
            ),
        )
        self.method_params = OmegaConf.to_container(config.method_params)
        self.counter_map: dict[str, Counter] = defaultdict(Counter)
        self.created_at = time.time()

    async def close(self):
        await self.client.aclose()

    def perf(self, **kwargs):
        elapsed = time.time() - self.created_at
        logger.info("------------E2E PERF-------------")
        logger.info(f"Time elapsed since ceated: {elapsed:.2f}s")
        for k in kwargs:
            logger.info(f"{k}: {kwargs[k]}")
        for k in self.counter_map:
            total = self.counter_map[k].total()
            logger.info(f"{k}: {total} ({total/elapsed:.2f}/s)")
        logger.info("-------------DETAIL--------------")
        for k,m in self.counter_map.items():
            logger.info(f"{k.upper()}")
            for p,v in m.items():
                logger.info(f"    {p}: {v}")
        


    async def _send_request(self, payload: Dict, purpose=None, method="generate") -> GenerateResponse:
        url = f"{self.server_url}/{method}"
        response = await self.client.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"request failed with status {response.status_code}: {response.text}")
        try:
            response_json = response.json()
            generate_response = GenerateResponse(**response_json)
            self.counter_map["requests"][purpose] += 1
            self.counter_map["sequences"][purpose] += len(generate_response.output_tokens)
            self.counter_map["input_tokens"][purpose] += generate_response.prompt_tokens
            self.counter_map["output_tokens"][purpose] += sum(generate_response.output_tokens)
            return generate_response
        except Exception as e:
            raise Exception(f"Failed to parse response: {e}")

    async def test(self):
        """
        Test the connection to the server.
        """
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
            "n": 1,
            "max_tokens": 10,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.7,
            "include_stop_str_in_output": True,
        }
        result = await self._send_request(payload, "test")
        logger.info(f"server connectted: {result}")

    async def search(self, messages: List[Dict[str, str]]) -> GenerateResponse:
        payload = {"messages": messages, **self.method_params["search"]}
        return await self._send_request(payload, "search")

    async def rollout(self, messages: List[Dict[str, str]]) -> GenerateResponse:
        payload = {"messages": messages, **self.method_params["rollout"]}
        return await self._send_request(payload, "rollout")

    async def reward(self, messages: List[Dict[str, str]]) -> GenerateResponse:
        payload = {"messages": messages, **self.method_params["reward"]}
        return await self._send_request(payload, "reward")

    async def cot_generate(self, messages: List[Dict[str, str]]) -> GenerateResponse:
        payload = {"messages": messages, **self.method_params["cot_generate"]}
        return await self._send_request(payload, "cot_generate")

class SimpleClient:
    def __init__(self):
        self.server_url = "http://127.0.0.1:8000/generate"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(5, read=300),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=10,
                keepalive_expiry=10,
            ),
        )

    async def _send_request(self, payload: Dict) -> GenerateResponse:
        response = await self.client.post(self.server_url, json=payload)
        if response.status_code != 200:
            raise Exception(f"request failed with status {response.status_code}: {response.text}")
        try:
            response_json = response.json()
            generate_response = GenerateResponse(**response_json)
            return generate_response
        except Exception as e:
            raise Exception(f"Failed to parse response: {e}")
    async def chat(self, message: str, **kwargs) -> GenerateResponse:
        payload = {"messages": [{"role": "user", "content": message}], **kwargs}
        return await self._send_request(payload)