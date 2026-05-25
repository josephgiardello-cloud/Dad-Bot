import unittest
from dadbot.infrastructure.device_state import Device, DeviceManager, Scheduler, SceneManager, parse_intent
import time

def dummy_action(context):
    context['called'] = True

class TestDeviceManager(unittest.TestCase):
    def setUp(self):
        self.dm = DeviceManager()
        self.device = Device('dev1', 'light', ['on', 'off'], {'power': 'off'})
        self.dm.devices['dev1'] = self.device

    def test_get_state(self):
        self.assertEqual(self.dm.get_state('dev1'), {'power': 'off'})
        self.assertIsNone(self.dm.get_state('unknown'))

    def test_set_state(self):
        # Should not raise, but is a stub
        self.dm.set_state('dev1', {'power': 'on'})

class TestScheduler(unittest.TestCase):
    def test_schedule_and_run(self):
        sched = Scheduler()
        context = {'called': False}
        sched.schedule(time.time() + 0.1, dummy_action, context)
        # Run in a thread for a short time
        import threading
        t = threading.Thread(target=sched.run_loop)
        t.start()
        time.sleep(0.2)
        sched.running = False
        t.join()
        self.assertTrue(context['called'])

class TestSceneManager(unittest.TestCase):
    def setUp(self):
        self.dm = DeviceManager()
        self.dm.devices['dev1'] = Device('dev1', 'light', ['on', 'off'], {'power': 'off'})
        self.scene = SceneManager(self.dm)

    def test_define_and_activate(self):
        self.scene.define_scene('morning', [('dev1', {'power': 'on'})])
        # set_state is a stub, but should not raise
        self.scene.activate('morning')

class TestParseIntent(unittest.TestCase):
    def test_parse_intent_stub(self):
        self.assertIsNone(parse_intent('turn on the light'))

if __name__ == '__main__':
    unittest.main()
