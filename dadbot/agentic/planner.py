
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable, Protocol, Union


class ToolProtocol(Protocol):
    def execute(self, action: str, **kwargs: Any) -> Any: ...

class ToolRegistryProtocol(Protocol):
    def get(self, name: str) -> Optional[ToolProtocol]: ...
    def describe_all(self) -> str: ...

LLMCallable = Union[Callable[[str], str], Callable[[str], Awaitable[str]]]


class ReActPlanner:
    def __init__(self, tool_registry: ToolRegistryProtocol, llm_port: LLMCallable) -> None:
        self.tools: ToolRegistryProtocol = tool_registry
        self.llm: LLMCallable = llm_port

    async def plan(self, user_input: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Run the async ReAct loop: build prompt, parse LLM output, execute tools, collect trace.
        Returns a dict with final answer and trace.
        """
        trace: List[Dict[str, Any]] = []
        question: str = user_input
        obs: Any = None
        for step in range(max_steps):
            prompt: str = self._build_prompt(question, trace)
            llm_output: str = await self._call_llm(prompt)
            thought, action = self._parse_llm_output(llm_output)
            obs = None
            if action:
                tool_name: Optional[str] = action.get("tool")
                args: Dict[str, Any] = action.get("args", {})
                tool = self.tools.get(tool_name) if tool_name else None
                if tool:
                    if asyncio.iscoroutinefunction(tool.execute):
                        obs = await tool.execute("run", **args)
                    else:
                        obs = tool.execute("run", **args)
                else:
                    obs = f"Tool '{tool_name}' not found."
            trace.append({
                "thought": thought,
                "action": action,
                "observation": obs
            })
            if self._is_final_answer(llm_output):
                break
        return {"trace": trace, "answer": obs}

    async def _call_llm(self, prompt: str) -> str:
        if asyncio.iscoroutinefunction(self.llm):
            return await self.llm(prompt)  # type: ignore
        return self.llm(prompt)  # type: ignore

    def _build_prompt(self, question: str, trace: List[Dict[str, Any]]) -> str:
        tool_descriptions: str = self.tools.describe_all()
        trace_str: str = "\n".join(
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
        thought: str = thought_match.group(1).strip() if thought_match else ""
        action: Optional[Dict[str, Any]] = None
        if action_match:
            try:
                action = json.loads(action_match.group(1))
            except Exception:
                action = None
        return thought, action

    def _is_final_answer(self, llm_output: str) -> bool:
        # Heuristic: if no Action block, treat as final answer
        return "Action:" not in llm_output