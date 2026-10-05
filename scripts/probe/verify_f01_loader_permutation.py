"""F01 독립 재현 — 라운드마다 로더를 새로 만들면 같은 순열을 보는가.

감사 F01 의 주장을 독립 확인한다. Ultralytics 를 import 하지 않고
그 코드가 하는 것과 같은 구성(상수 시드 generator + RandomSampler + _RepeatSampler)을
직접 만들어 본다. GPU 를 쓰지 않는다.

- 연합: 라운드마다 트레이너·로더 신규 → generator 가 매번 상수로 초기화
- 중앙집중: 로더 1개로 N epoch 연속 → generator 가 계속 소비됨
"""
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

CONST_SEED = 6148914691236517205  # ultralytics/data/build.py:365, RANK=-1 이면 +(-1)
RANK = -1
N = 64
BATCH = 8


class Toy(Dataset):
    def __len__(self): return N
    def __getitem__(self, i): return i


class _RepeatSampler:
    """ultralytics/data/build.py 의 무한 반복 샘플러와 같은 구조."""
    def __init__(self, sampler): self.sampler = sampler
    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def make_loader():
    """build_dataloader 와 같은 방식: 호출마다 새 Generator + 상수 시드."""
    g = torch.Generator()
    g.manual_seed(CONST_SEED + RANK)
    ds = Toy()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                        generator=g, drop_last=False)
    # InfiniteDataLoader 처럼 반복자를 붙들어 둔다
    object.__setattr__(loader, "batch_sampler", _RepeatSampler(loader.batch_sampler))
    return loader


def epochs_from(loader, n_epochs):
    """연속한 n_epochs 개의 순열을 뽑는다."""
    it = iter(loader)
    n_batches = (N + BATCH - 1) // BATCH
    out = []
    for _ in range(n_epochs):
        ep = []
        for _ in range(n_batches):
            ep.extend(int(x) for x in next(it))
        out.append(ep)
    return out


print("=" * 70)
print("F01 재현 — 상수 시드 generator 가 라운드 간 순열을 고정하는가")
print("=" * 70)

# (1) 연합 흉내: 라운드마다 로더를 새로 만들어 각각 E=2 epoch
round1 = epochs_from(make_loader(), 2)
round2 = epochs_from(make_loader(), 2)
round9 = epochs_from(make_loader(), 2)

print("\n[연합 흉내 — 라운드마다 로더 신규, 각 2 epoch]")
print(f"  라운드1 epoch1 앞 12개: {round1[0][:12]}")
print(f"  라운드2 epoch1 앞 12개: {round2[0][:12]}")
print(f"  라운드9 epoch1 앞 12개: {round9[0][:12]}")
print(f"  라운드1 == 라운드2 ? {round1 == round2}")
print(f"  라운드1 == 라운드9 ? {round1 == round9}")

# (2) 중앙집중 흉내: 로더 1개로 6 epoch 연속
central = epochs_from(make_loader(), 6)
print("\n[중앙집중 흉내 — 로더 1개로 6 epoch 연속]")
for i, ep in enumerate(central, 1):
    print(f"  epoch{i} 앞 12개: {ep[:12]}")
uniq = len({tuple(e) for e in central})
print(f"  6 epoch 중 서로 다른 순열 수: {uniq}")

# (3) 핵심 대조: 연합이 50 라운드 도는 동안 보는 서로 다른 순열 수
fed_perms = set()
for r in range(50):
    for ep in epochs_from(make_loader(), 2):
        fed_perms.add(tuple(ep))
cen_perms = {tuple(e) for e in epochs_from(make_loader(), 100)}

print("\n" + "=" * 70)
print("[핵심 대조] R=50 · E=2 = 100 epoch 을 각각 어떻게 소비하는가")
print("=" * 70)
print(f"  연합(라운드마다 새 로더): 서로 다른 순열 {len(fed_perms)} 개 / 100 epoch")
print(f"  중앙집중(로더 1개 연속) : 서로 다른 순열 {len(cen_perms)} 개 / 100 epoch")
print()
if len(fed_perms) < len(cen_perms):
    print("  → F01 성립. 같은 '100 epoch' 이라도 연합이 보는 데이터 순서 다양성이 낮다.")
else:
    print("  → F01 반증. 두 경로의 순열 다양성이 같다.")

# (4) 재시드를 켜면 달라지는가 (LoaderReseed 가 하는 일)
def make_loader_reseeded(seed, epoch):
    g = torch.Generator()
    g.manual_seed(seed + 1_000_003 * epoch)
    ds = Toy()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                        generator=g, drop_last=False)
    object.__setattr__(loader, "batch_sampler", _RepeatSampler(loader.batch_sampler))
    return loader

reseeded = set()
for r in range(50):
    for e in range(2):
        global_epoch = r * 2 + e
        reseeded.add(tuple(epochs_from(make_loader_reseeded(20260828, global_epoch), 1)[0]))
print(f"\n  참고 — epoch 기반 재시드를 켜면: 서로 다른 순열 {len(reseeded)} 개 / 100 epoch")
