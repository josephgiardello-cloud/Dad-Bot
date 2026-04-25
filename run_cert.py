import logging, sys
from pathlib import Path
from tempfile import TemporaryDirectory
from tests.stress.phase4_certification_gate import Phase4CertificationGate, build_bot
logging.basicConfig(level=logging.WARNING)
with TemporaryDirectory() as tmp:
    bot = build_bot(Path(tmp))
    gate = Phase4CertificationGate(bot)
    report = gate.run_all(long_horizon_turns=10, memory_growth_turns=10, replay_turns=5, concurrency_threads=4)
    bot.shutdown()
print('certification:', report['phase4_certification'])
print('score:', report['score'])
print('failures:', report['failures'][:5])
print('risk_flags:', report['risk_flags'][:5])
for name, r in report['results'].items():
    p = r['passed']
    s = r['score']
    d = r['duration_s']
    f = len(r['failures'])
    print(f'  {name}: passed={p} score={s} dur={d:.1f}s failures={f}')
