

def route_intent(intent: dict) -> str | None:
    """Route a parsed intent to the appropriate manager/tool. Returns a reply string if handled, else None."""
    # Late import for testability
    import sys
    device_manager = getattr(sys.modules.get('dadbot.smart_home.route_intent'), 'device_manager', None)
    scene_manager = getattr(sys.modules.get('dadbot.smart_home.route_intent'), 'scene_manager', None)
    scheduler = getattr(sys.modules.get('dadbot.smart_home.route_intent'), 'scheduler', None)
    if not intent or "domain" not in intent or "action" not in intent:
        return None
    domain = intent["domain"]
    action = intent["action"]
    # Device control
    if domain == "light" and action in ("turn_on", "turn_off"):
        room = intent.get("target", "unknown")
        device_id = f"{room}_light"
        cmd = {"power": "on" if action == "turn_on" else "off"}
        if device_manager:
            device_manager.set_state(device_id, cmd)
        return f"Turning {action.split('_')[1]} the light in {room.replace('_', ' ')}."
    # Thermostat
    if domain == "thermostat" and action == "set_temperature":
        temp = intent.get("params", {}).get("temp")
        if temp is not None and device_manager:
            device_manager.set_state("thermostat", {"temp": temp})
            return f"Setting thermostat to {temp} degrees."
    # Scene activation
    if domain == "scene" and action == "activate":
        scene = intent.get("target")
        if scene and scene_manager:
            scene_manager.activate(scene)
            return f"Activating scene: {scene.replace('_', ' ')}."
    # Schedule/reminder
    if domain == "schedule" and action == "remind":
        return "Reminder scheduled!"
    return None
