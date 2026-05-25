import unittest
from dadbot.smart_home.intent_parser import parse_intent
from dadbot.smart_home.route_intent import route_intent
from dadbot.infrastructure.device_state import DeviceManager, Device

class TestIntentIntegration(unittest.TestCase):
    def setUp(self):
        # Setup a device manager with a test device
        self.dm = DeviceManager()
        self.dm.devices['living_room_light'] = Device('living_room_light', 'light', ['on', 'off'], {'power': 'off'})
        self.dm.devices['thermostat'] = Device('thermostat', 'thermostat', ['set_temp'], {'temp': 70})
        # Patch the singleton used by route_intent
        import dadbot.smart_home.route_intent as route_mod
        route_mod.device_manager = self.dm
        route_mod.scene_manager = None
        route_mod.scheduler = None

    def test_parse_and_route_turn_on_light(self):
        intent = parse_intent('Turn on the light in living room')
        self.assertIsNotNone(intent)
        self.assertEqual(intent['domain'], 'light')
        reply = route_intent(intent)
        self.assertIn('Turning on the light', reply)
        self.assertEqual(self.dm.devices['living_room_light'].state['power'], 'on')

    def test_parse_and_route_set_thermostat(self):
        intent = parse_intent('Set temperature to 72')
        self.assertIsNotNone(intent)
        self.assertEqual(intent['domain'], 'thermostat')
        reply = route_intent(intent)
        self.assertIn('Setting thermostat to 72', reply)
        self.assertEqual(self.dm.devices['thermostat'].state['temp'], 72)

    def test_parse_and_route_unknown(self):
        intent = parse_intent('Do something random')
        self.assertIsNone(intent)

if __name__ == '__main__':
    unittest.main()
