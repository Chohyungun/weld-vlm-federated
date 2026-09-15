"""F04 독립 재현: 어댑터 훅(weight 만)을 준 ResumeCheckpointer.save() 가 훅을 부르는가, 저장 키가 무엇인가."""
import sys, tempfile, pathlib
import torch
from detection.resume import ResumeCheckpointer, latest_resume
from tests.test_detection_resume import _FakeTrainer, _Net, _identity
calls = []
def hook(tr):
    calls.append(1)
    return {"lin.weight": tr.model.lin.weight}
d = pathlib.Path(tempfile.mkdtemp())
tr = _FakeTrainer(_Net(seed=1), epoch=4, start_epoch=4)
ck = ResumeCheckpointer(d, identity=_identity(), state_dict_fn=hook)
ck.save(tr)
st = latest_resume(d)
full_keys = list(tr.model.state_dict().keys())
print(f"tree={sys.argv[1]} | hook calls={len(calls)} | saved keys={st.payload['canonical_keys']} | model keys={full_keys}")
