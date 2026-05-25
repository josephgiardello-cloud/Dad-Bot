import re

def extract_room(text: str) -> str:
    # Simple room extractor for demo
    for room in ["living room", "kitchen", "bedroom", "bathroom", "office"]:
        if room in text:
            return room.replace(" ", "_")
    return "unknown"

def extract_number(text: str) -> int:
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0

def parse_intent(text: str) -> dict | None:
    text = text.lower()
    if "turn on" in text and "light" in text:
        return {"domain": "light", "action": "turn_on", "target": extract_room(text), "confidence": 1.0}
    if "turn off" in text and "light" in text:
        return {"domain": "light", "action": "turn_off", "target": extract_room(text), "confidence": 1.0}
    if "set temperature to" in text:
        temp = extract_number(text)
        return {"domain": "thermostat", "action": "set_temperature", "params": {"temp": temp}, "confidence": 1.0}
    if "activate" in text and "scene" in text:
        scene = text.split("activate")[-1].strip().replace(" ", "_")
        return {"domain": "scene", "action": "activate", "target": scene, "confidence": 1.0}
    if "remind me" in text:
        # Very basic schedule intent
        return {"domain": "schedule", "action": "remind", "params": {"text": text}, "confidence": 0.9}
    return None
