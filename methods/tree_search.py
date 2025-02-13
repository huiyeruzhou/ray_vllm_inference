from typing import List, Optional, Dict
from collections import Counter
from api import APIClient
import asyncio
import logging
import json
logger = logging.getLogger(__name__)


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
        self.evaluate_result: Optional[List[Dict[str, str]]] = None
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
                messages[-1]["content"] = current.value + messages[-1]["content"]
            else:
                messages.append({"role": current.role, "content": current.value})
            current = current.parent
        if messages and messages[0]["role"] == "assistant":
            messages[0]["content"]
        return messages[::-1]  # 从根节点开始

    def __repr__(self):
        return f"TreeNode(value={self.value}, role={self.role}, score={self.score})"

    def to_dict(self):
        return {
            "value": self.value,
            "score": self.score,
            "evaluate_result": self.evaluate_result,
            "role": self.role,
            "children": [child.to_dict() for child in self.children],
        }


async def tree_search(
    api_client: APIClient,
    client_id: int,
    output_dir: str,
    data: str,
    *,
    depth: int,
    beam: int,
) -> Dict[str, int]:
    """
    perform beam search in given root, return root node, best trajectory, and token counter.
    """
    root_node = TreeNode(data["prompt"], role="user")
    current_level = [root_node]
    token_counter: Dict[str, int] = Counter()
    for _ in range(depth - 1):
        trajectories = [node.trajectory for node in current_level]
        try:
            search_responses = await asyncio.gather(
                *[api_client.search(traj) for traj in trajectories],
            )
        except Exception as e:
            logger.error(f"Error during search requests: {e}")
            raise

        next_level_nodes = []
        async with asyncio.TaskGroup() as tg:
            for node, result in zip(current_level, search_responses):
                token_counter["input_tokens"] += result.prompt_tokens
                token_counter["output_tokens"] += sum(result.output_tokens)
                for generated_text in result.texts:
                    new_child = TreeNode(generated_text, role="assistant", parent=node)
                    next_level_nodes.append(new_child)
                    
                    tg.create_task(evaluate_fn(api_client, new_child, data))
               

        next_level_nodes.sort(key=lambda x: x.score, reverse=True)
        current_level = next_level_nodes[:beam]

    # rollout for the best node
    try:
        rollout_response = await api_client.rollout(current_level[0].trajectory)
        token_counter["input_tokens"] += rollout_response.prompt_tokens
        token_counter["output_tokens"] += sum(rollout_response.output_tokens)
        final_result = TreeNode(rollout_response.texts[0], role="assistant", parent=current_level[0])
        with open(f"{output_dir}/{client_id}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"tree": root_node.to_dict(), "result": final_result.trajectory[-1]["content"]},
                f,
                indent=4,
                ensure_ascii=False,
            )
    except Exception as e:
        logger.error(f"Rollout request failed for node '{current_level[0].value}': {e}")
        raise
    return token_counter


async def evaluate_fn(api_client: APIClient, node: TreeNode, data: dict) -> float:
    traj_parent = node.parent.trajectory
    try:
        prompt = f"""You are an expert in evaluating math solutions. Assess the following next step based on correctness, relevance, and clarity, and assign a score from 1 (poor) to 5 (perfect). Repititions, irrelevance, or wrong should be penalized. Deeply thought such as reflection, planning, turning into a better way of thinking should be rewarded.

### Problem:
{data['problem']}

### Current Step:
{traj_parent[1]['content'] if len(traj_parent) > 1 else 'NONE'}

### Next Step:
{node.value}

Provide your response strictly in this JSON format:
```json
{{
    "score": <1-5>
}}
```
"""
        evaluate_response = await api_client.reward([{"role": "user", "content": prompt}])
        score_text = evaluate_response.texts[0]
        import re
        json_text = re.findall(r"```json(.*?)```", score_text, re.DOTALL)
        if len(json_text) == 0:
            raise ValueError(f"Wrong JSON: {len(json_text)}")
        json_text = json_text[-1]
        score_json = json.loads(json_text)
        correctness = score_json["score"]
        return correctness, [{"role": "user", "content": prompt}, {"role": "assistant", "content": score_text}]
    except Exception as e:
        logger.error(f"Failed to evaluate trajectory: {e}")
        if "score_text" in locals():
            logger.error(f"Failed to evaluate trajectory. Score text:\n==================\n{score_text}\n================")
        return -1, [{"role": "user", "content": prompt}, {"role": "assistant", "content": score_text}]
