#!/usr/bin/env bash
# 총괄 머지 게이트 — 테스트가 실패하면 머지가 물리적으로 불가능하게 한다.
#
# 배경: 6주간 게이트가 자기신고로 돌았고, 한 번은 `pytest | tail && git merge` 의
# 파이프가 종료 코드를 삼켜 테스트 실패 상태에서 머지가 통과했다(79번 C7·총괄 오류).
# 이 스크립트는 그 경로를 막는다: 머지를 --no-commit 으로 열고, 그 트리에서 전체
# 테스트를 돌리고, 통과해야만 커밋한다. 실패하면 머지를 되돌리고 0이 아닌 코드로 나간다.
#
#   scripts/cto_gate.sh <branch> <commit-msg-file>
set -u
BRANCH="${1:?브랜치를 지정하라}"
MSGFILE="${2:?커밋 메시지 파일을 지정하라}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "!! 작업 트리가 깨끗하지 않다. 게이트를 열 수 없다." >&2
  exit 2
fi

# 게이트 잠금 — 게이트가 열려 있는 동안 다른 커밋이 머지를 조기 종결시키는 사고 방지
# (실사고 2026-09-03: 열린 머지 중 총괄 문서 커밋이 F 머지를 테스트 완료 전에 완결시켰다)
LOCK=".git/CTO_GATE_OPEN"
if [ -f "$LOCK" ]; then
  echo "!! 다른 게이트가 열려 있다 ($LOCK). 동시 게이트 금지." >&2
  exit 5
fi
echo "$BRANCH $(date -u +%FT%TZ)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

git merge --no-commit --no-ff "$BRANCH"
MERGE_RC=$?
if [ $MERGE_RC -ne 0 ]; then
  echo "!! 머지 충돌 — 수동 해소 후 다시." >&2
  git merge --abort 2>/dev/null
  exit 3
fi

echo "== 게이트: 전체 테스트 =="
# GATE_DEFER_HEAVY=1: 등록된 장기 실행(본실험 학습)이 자원을 점유한 동안,
# resource_heavy 마커 시험을 "건너뛰기"가 아니라 **유예 원장에 기록**하고 나머지 전량을 돌린다.
# 유예분은 다음 유휴 창에서 반드시 재실행한다 — 원장이 그 의무의 증거다.
if [ "${GATE_DEFER_HEAVY:-0}" = "1" ]; then
  uv run pytest -q -m "not resource_heavy"
  TEST_RC=$?
  DEFERRED=$(uv run pytest -q -m resource_heavy --collect-only 2>/dev/null | grep -c "::" || echo 0)
  # 원장 기록을 머지 커밋에 원자 포함 — 커밋 뒤에 쓰면 다음 게이트가 더러운 트리를 본다(실사고 09-03)
  echo "$(date -u +%FT%TZ) merge=$BRANCH deferred=$DEFERRED reason=main-experiment-running" >> docs/dev_log/gate_deferred_ledger.txt
  git add docs/dev_log/gate_deferred_ledger.txt
  echo "== 유예 원장 기록: resource_heavy $DEFERRED 건 (본실험 점유 중, 머지 커밋에 포함) =="
else
  uv run pytest -q
  TEST_RC=$?
fi
if [ $TEST_RC -ne 0 ]; then
  echo "!! 테스트 실패 (exit $TEST_RC) — 머지를 되돌린다." >&2
  git merge --abort 2>/dev/null || git reset --merge
  exit 4
fi

CTO_GATE_COMMIT=1 git commit -F "$MSGFILE" --no-edit
echo "== 게이트 통과, 머지 완료: $(git log --oneline -1) =="
