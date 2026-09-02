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

git merge --no-commit --no-ff "$BRANCH"
MERGE_RC=$?
if [ $MERGE_RC -ne 0 ]; then
  echo "!! 머지 충돌 — 수동 해소 후 다시." >&2
  git merge --abort 2>/dev/null
  exit 3
fi

echo "== 게이트: 전체 테스트 =="
uv run pytest -q
TEST_RC=$?
if [ $TEST_RC -ne 0 ]; then
  echo "!! 테스트 실패 (exit $TEST_RC) — 머지를 되돌린다." >&2
  git merge --abort 2>/dev/null || git reset --merge
  exit 4
fi

git commit -F "$MSGFILE" --no-edit
echo "== 게이트 통과, 머지 완료: $(git log --oneline -1) =="
