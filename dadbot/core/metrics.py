"""
Structured metrics collection for DadBot.
Provides counters, timers, and gauges for reliability and performance monitoring.
"""
import time
from collections import defaultdict

class Metrics:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.gauges = {}

    def inc(self, name, value=1):
        self.counters[name] += value

    def observe(self, name, value):
        self.timers[name].append(value)

    def set_gauge(self, name, value):
        self.gauges[name] = value

    def get_metrics(self):
        return {
            'counters': dict(self.counters),
            'timers': {k: (min(v), max(v), sum(v)/len(v) if v else 0) for k, v in self.timers.items()},
            'gauges': dict(self.gauges),
        }

metrics = Metrics()
