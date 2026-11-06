"""Provisional chat delivery, isolated from committed history and speech."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from queue import Empty, Queue

draft_chunk_callback: ContextVar[Callable[[str], None] | None] = ContextVar(
    "draft_chunk_callback", default=None,
)


def run_with_chat_drafts(
    run_turn: Callable[[], tuple[str, bool]],
    render_draft: Callable[[str], None],
) -> tuple[str, bool]:
    """Run a turn off-thread; render drafts and return its final result on the caller thread."""
    chunks: Queue[str] = Queue()

    def worker() -> tuple[str, bool]:
        token = draft_chunk_callback.set(chunks.put)
        try:
            return run_turn()
        finally:
            draft_chunk_callback.reset(token)

    context = copy_context()
    parts: list[str] = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dad-chat") as pool:
        future = pool.submit(context.run, worker)
        while not future.done() or not chunks.empty():
            try:
                parts.append(chunks.get(timeout=0.05))
            except Empty:
                continue
            render_draft("".join(parts))
        return future.result()
