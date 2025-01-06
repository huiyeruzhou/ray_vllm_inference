# client.py

import asyncio
import uvloop
import json
import time
from typing import Optional, List, Dict, Any
from collections import Counter

import httpx
from omegaconf import DictConfig, OmegaConf
import hydra
from ray_vllm_inference.protocol import GenerateResponse

# use event loop of uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import logging
logger = logging.getLogger(__name__)


import httpx


class APIClient:
    """
    A simple API client for different types of generate requests.
    """
    def __init__(self, config):
        self.server_url = config.url
        if self.server_url.endswith("/"):
            self.server_url = self.server_url[:-1]
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(5, read=config.read_timeout),
                                        limits=httpx.Limits(max_connections=config.max_connections,
                                                         max_keepalive_connections=config.max_keepalive_connections,
                                                         keepalive_expiry=config.keepalive_expiry))

    async def close(self):
        await self.client.aclose()

    async def _send_request(self, payload: Dict, method="generate") -> GenerateResponse:
        url = f"{self.server_url}/{method}"
        response = await self.client.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"request failed with status {response.status_code}: {response.text}")
        try:
            response_json = response.json()
            generate_response = GenerateResponse(**response_json)
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
        }
        result =  await self._send_request(payload)
        logger.info(f"server connectted: {result}")
        


    async def search(self, messages: List[Dict[str, str]]) -> GenerateResponse:
        payload = {
            "messages": messages,
            "n": 5,
            "max_tokens": 512,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.7,
            "stop": ["\n\n", "<|EOS_TOKEN|>"],
        }
        return await self._send_request(payload)

    async def rollout(self, messages: List[Dict[str, str]]) -> GenerateResponse:
        payload = {
            "messages": messages,
            "n": 1,
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.7,
            "stop": ["<|EOS_TOKEN|>"],
        }
        return await self._send_request(payload)

    async def reward(self, messages: List[Dict[str, str]]) -> GenerateResponse:
        payload = {
            "messages": messages,
            "n": 1,
            "max_tokens": 2048,
            "temperature": 0.4,
            "top_k": -1,
            "top_p": 1.0,
            "stop": ["<|EOS_TOKEN|>"],
        }
        return await self._send_request(payload)

# 定义 TreeNode 类
class TreeNode:
    def __init__(self, value: str, role: str = "user", parent: Optional["TreeNode"] = None):
        self.value = value
        self.role = role
        self.parent = parent
        if parent:
            parent.add_child(self)
        self.children: List["TreeNode"] = []
        self.score = 0.0
        self._trajectory: Optional[List[Dict[str, str]]] = None

    def add_child(self, child: "TreeNode"):
        self.children.append(child)

    @property
    def trajectory(self) -> List[Dict[str, str]]:
        if self._trajectory is None:
            self._trajectory = self._generate_trajectory()
        return self._trajectory

    def _generate_trajectory(self) -> List[Dict[str, str]]:
        messages = []
        current = self
        while current:
            if messages and messages[-1]["role"] == current.role:
                messages[-1]['content'] = current.value + "\n\n" + messages[-1]["content"]
            else:
                messages.append({"role": current.role, "content": current.value})
            current = current.parent
        if messages and messages[0]["role"] == "assistant":
            messages[0]['content'] += "\n\n"
        return messages[::-1]  # 从根节点开始

    def __repr__(self):
        return f"TreeNode(value={self.value}, role={self.role}, score={self.score})"

    def to_dict(self):
        return {
            "value": self.value,
            "score": self.score,
            "role": self.role,
            "children": [child.to_dict() for child in self.children],
        }



async def tree_search(api_client: APIClient, root_node: TreeNode, depth: int, evaluate_fn) -> tuple[TreeNode, Dict[str, int]]:
    """
    perform beam search in given root, return root node, best trajectory, and token counter.
    """
    current_level = [root_node]
    token_counter: Dict[str, int] = Counter()
    for _ in range(depth - 1):
        trajectories = [node.trajectory for node in current_level]
        try:
            search_responses = await asyncio.gather(
                *[api_client.search(traj) for traj in trajectories],
            )
        except Exception as e:
            print(f"Error during search requests: {e}")
            raise

        next_level_nodes = []
        for node, result in zip(current_level, search_responses):
            token_counter["input_tokens"] += result.prompt_tokens
            token_counter["output_tokens"] += sum(result.output_tokens)
            for generated_text in result.texts:
                new_child = TreeNode(generated_text, role="assistant", parent=node)
                next_level_nodes.append(new_child)
                try:
                    score_response = await evaluate_fn(api_client, new_child.trajectory)
                    new_child.score = score_response
                except Exception as e:
                    print(f"Evaluation failed for node '{new_child.value}': {e}")
                    new_child.score = -1 

        next_level_nodes.sort(key=lambda x: x.score, reverse=True)
        current_level = next_level_nodes[:5]

    # rollout for the best node
    if current_level:
        try:
            rollout_response = await api_client.rollout(current_level[0].trajectory)
            token_counter["input_tokens"] += rollout_response.prompt_tokens
            token_counter["output_tokens"] += sum(rollout_response.output_tokens)
            for generated_text in rollout_response.texts:
                new_child = TreeNode(generated_text, role="assistant", parent=current_level[0])
        except Exception as e:
            print(f"Rollout request failed for node '{current_level[0].value}': {e}")
    return root_node, new_child, token_counter


async def evaluate_fn(api_client: APIClient, trajectory: List[Dict[str, str]]) -> float:
    return len(trajectory)
    try:
        evaluate_response = await api_client.reward(trajectory)
        score_text = evaluate_response.texts[0]
        score_json = json.loads(score_text)
        correctness = score_json.get("correctness", 1)
        contribution = score_json.get("contribution", 1)
        simplicity = score_json.get("simplicity", 1)
        return (correctness + contribution + simplicity) / 3
    except Exception as e:
        print(f"Failed to evaluate trajectory: {e}")
        if 'score_text' in locals():
            print(f"Failed to evaluate trajectory. Score text: {score_text}")
        return -1


async def client_request(api_client: APIClient, client_id: int, root_node: TreeNode):
    try:
        start_time = time.time()
        root, traj, token_counter = await tree_search(api_client, root_node, depth=5, evaluate_fn=evaluate_fn)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(
            f"""Client {client_id} received result (Time taken: {time_elapsed:.2f}s). Input tokens: {token_counter.get('input_tokens',0)}, Output tokens: {token_counter.get('output_tokens',0)}."""
        )

        with open(f"outputs/tree/{client_id}.json", "w", encoding="utf-8") as f:
            json.dump({"tree": root.to_dict(), "traj": traj}, f, indent=4, ensure_ascii=False)
        return token_counter
    except asyncio.CancelledError:
        raise
    except TimeoutError as e:
        print(f"Client {client_id} timed out: {e}")
    except Exception as e:
        print(f"Client {client_id} encountered an error: {e}")
        raise


def load_dataset(config):
    with open(config.dataset.path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    return ["""Solve the following math problem step by step. Use double newlines as the end of each step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:". 

Here your step by step response:
""".format(Question=d["problem"])
        for d in data[: config.dataset.max_length]]


async def main_async(config: DictConfig):
    OmegaConf.resolve(config)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    # load dataset, create root nodes
    initial_nodes = load_dataset(config)
    root_nodes = [TreeNode(value=node) for node in initial_nodes]

    # prepare output directory
    import os
    os.makedirs("outputs/tree", exist_ok=True)
    print(f"Output directory: {os.path.join(os.getcwd(), 'outputs')}")

    # make sure the server is connected
    api_client = APIClient(config=config.http)
    try:
        await api_client.test()
    except Exception as e:
        print(f"Test request failed: {e}")
        raise

    # async tasks
    # try:
    tasks: list[asyncio.Task] = []
    merged_counter: Dict[str, int] = Counter()
    try:
        async with asyncio.TaskGroup() as tg:
            for i, root_node in enumerate(root_nodes):
                t = tg.create_task(client_request(api_client, i, root_node))
                t.add_done_callback(lambda t: merged_counter.update(t.result()))
            start_time = time.time()
        end_time = time.time()
        time_elapsed = end_time - start_time

        print(
            f"""Total time taken: {time_elapsed:.2f}s.
    Total input tokens: {merged_counter.get('input_tokens',0)}, speed {merged_counter.get('input_tokens',0)/(time_elapsed):.2f} tokens/s, 
    Total output tokens: {merged_counter.get('output_tokens',0)}, speed {merged_counter.get('output_tokens',0)/(time_elapsed):.2f} tokens/s"""
        )
    except asyncio.CancelledError:
        logger.error("Main task was cancelled. Cancelling all client tasks...")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await api_client.close()


if __name__ == "__main__":
    @hydra.main(config_path="conf", config_name="client", version_base=None)
    def main(cfg: DictConfig):
        try:
            asyncio.run(main_async(cfg))
        except KeyboardInterrupt:
            print("Interrputed by user (Ctrl+C)")
    main()
