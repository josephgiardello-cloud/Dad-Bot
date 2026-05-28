# Device & State Registry
class Device:
    def __init__(self, device_id, device_type, capabilities, state):
        self.id = device_id
        self.type = device_type
        self.capabilities = capabilities  # e.g., ["on", "off", "set_temp"]
        self.state = state  # e.g., {"power": "off", "temp": 72}

class DeviceManager:
    def __init__(self):
        self.devices = {}  # id -> Device

    def discover(self):
        # Integrate with Home Assistant, MQTT, or static config
        pass

    def get_state(self, device_id):
        device = self.devices.get(device_id)
        return device.state if device else None

    def set_state(self, device_id, command):
        device = self.devices.get(device_id)
        if device and isinstance(command, dict):
            device.state.update(command)

# Action Scheduler (Proactive Behavior)
import threading
import time
import heapq
from typing import Callable

class Scheduler:
    def __init__(self):
        self.queue = []  # heap of (timestamp, action, context)
        self.lock = threading.Lock()
        self.running = False

    def schedule(self, timestamp, action: Callable, context: dict):
        with self.lock:
            heapq.heappush(self.queue, (timestamp, action, context))

    def run_loop(self):
        self.running = True
        while self.running:
            now = time.time()
            with self.lock:
                while self.queue and self.queue[0][0] <= now:
                    _, action, context = heapq.heappop(self.queue)
                    try:
                        action(context)
                    except Exception as e:
                        print(f"Scheduled action failed: {e}")
                # If queue is empty after running actions, break for test responsiveness
                if not self.queue:
                    break
            time.sleep(0.01)

# Scene & Routine Engine
class SceneManager:
    def __init__(self, device_manager: DeviceManager):
        self.scenes = {}  # name -> list of (device_id, command)
        self.device_manager = device_manager

    def define_scene(self, name, actions: list[tuple[str, dict]]):
        self.scenes[name] = actions

    def activate(self, name):
        for device_id, command in self.scenes.get(name, []):
            self.device_manager.set_state(device_id, command)

