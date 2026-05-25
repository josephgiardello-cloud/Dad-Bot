from typing import Any, Dict, List, Optional, Tuple

class ReActPlanner:
    def __init__(self, tool_registry, llm_port):
        self.tools = tool_registry
        self.llm = llm_port

    def plan(self, user_input: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Run the ReAct loop: build prompt, parse LLM output, execute tools, collect trace.
        Returns a dict with final answer and trace.
        """
        trace: List[Dict[str, Any]] = []
        question = user_input
        for step in range(max_steps):
            prompt = self._build_prompt(question, trace)
            llm_output = self.llm(prompt)
            thought, action = self._parse_llm_output(llm_output)
            obs = None
            if action:
                tool_name = action.get("tool")
                args = action.get("args", {})
                tool = self.tools.resolve(tool_name)
                obs = tool.executor(**args)
            trace.append({
                "thought": thought,
                "action": action,
                "observation": obs
            })
            if self._is_final_answer(llm_output):
                break
        return {"trace": trace, "answer": obs}

    def _build_prompt(self, question: str, trace: List[Dict[str, Any]]) -> str:
        tool_descriptions = self.tools.describe_all()
        trace_str = "\n".join(
            f"Thought: {t['thought']}\nAction: {t['action']}\nObservation: {t['observation']}" for t in trace
        )
        return f"""You are DadBot. Answer the following question. You have these tools:
{tool_descriptions}

Use the format:
Thought: ...
Action: {{"tool": "tool_name", "args": {{...}} }}
Observation: ...
... (repeat)

Question: {question}
{trace_str}
"""

    def _parse_llm_output(self, llm_output: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        # Simple regex-based parse for Thought/Action
        import re, json
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

    def _is_final_answer(self, llm_output: str) -> bool:
        # Heuristic: if no Action block, treat as final answer
        return "Action:" not in llm_output