import re
import json
from typing import Any, Dict, Optional, Tuple

def parse_react_output(llm_output: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parse LLM output for ReAct loop. Returns (thought, action_dict or None).
    """
    thought_match = re.search(r"Thought: (.*)", llm_output)
    action_match = re.search(r"Action: (\{.*\})", llm_output)
    thought = thought_match.group(1).strip() if thought_match else ""
    action = None
    if action_match:
        try:
            action = json.loads(action_match.group(1))
        except Exception:
            action = None
    return thought, action
