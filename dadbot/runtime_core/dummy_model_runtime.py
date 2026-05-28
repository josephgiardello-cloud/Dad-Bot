import json
import urllib.request
import socket

class DummyModelRuntime:
    def initialize_tokenizer(self, model_name=None):
        # Return a dummy tokenizer or None
        return None

    def current_tokenizer(self, model_name=None):
        return None

    def model_chars_per_token_estimate(self, model_name=None):
        return 4.0

    def ollama_status(self):
        """Check if Ollama server is running and return status dict for UI."""
        status = {"connected": False, "models": [], "error": None}
        url = "http://localhost:11434/api/tags"
        try:
            # Try to connect to the Ollama server quickly (timeout 0.5s)
            with socket.create_connection(("localhost", 11434), timeout=0.5):
                pass
        except Exception as e:
            status["error"] = f"Ollama not reachable: {e}"
            return status
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                status["connected"] = True
                status["models"] = models
        except Exception as e:
            status["error"] = f"Ollama API error: {e}"
        return status
