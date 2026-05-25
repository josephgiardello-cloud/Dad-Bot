from dadbot.base.tool_base import Tool
from typing import Any
import os

class FileSystemTool(Tool):
    def __init__(self):
        super().__init__(
            name="file_read",
            description="Read the contents of a file from the local file system.",
            parameters={"path": "Path to the file to read (str)"}
        )

    def run(self, **kwargs) -> Any:
        path = kwargs.get("path")
        if not path or not os.path.isfile(path):
            return {"error": "File not found"}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {"content": f.read()}
        except Exception as e:
            return {"error": str(e)}
