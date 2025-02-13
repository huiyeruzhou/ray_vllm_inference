from typing import Dict
from collections import Counter
from api import APIClient
import logging
import json
logger = logging.getLogger(__name__)


async def cot(
    api_client: APIClient,
    client_id: int,
    output_dir: str,
    data: str,
    **kwargs
) -> Dict[str, int]:
    """
    perform beam search in given root, return root node, best trajectory, and token counter.
    """
    token_counter: Dict[str, int] = Counter()
    try:
        rollout_response = await api_client.cot_generate([{"role": "user", "content": data['prompt']}])
        token_counter["input_tokens"] = rollout_response.prompt_tokens
        token_counter["output_tokens"] = rollout_response.output_tokens[0]
        final_result = rollout_response.texts[0]
        with open(f"{output_dir}/{client_id}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"result": final_result},
                f,
                indent=4,
                ensure_ascii=False,
            )
    except Exception as e:
        logger.error(f"Cot Generate request failed")
        raise
    return token_counter

