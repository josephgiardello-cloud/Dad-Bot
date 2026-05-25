"""
Smart home MQTT client integration for DadBot.
Handles connection, subscription, and publishing to smart home devices.
"""
import paho.mqtt.client as mqtt

class SmartHomeMQTTClient:
    def __init__(self, broker_url: str, broker_port: int = 1883):
        self.client = mqtt.Client()
        self.broker_url = broker_url
        self.broker_port = broker_port

    def connect(self):
        self.client.connect(self.broker_url, self.broker_port)
        self.client.loop_start()

    def subscribe(self, topic: str):
        self.client.subscribe(topic)

    def publish(self, topic: str, payload: str):
        self.client.publish(topic, payload)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
