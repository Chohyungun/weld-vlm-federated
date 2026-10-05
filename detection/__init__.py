"""분리형 검출 학습 (로컬·중앙·연합 3칸).

세 칸이 모두 `round_runner.train_round` 하나를 통과한다. 로컬·중앙은 `R=1, E=N`인
퇴화 케이스다.

이 파일은 비워 둔다. `detection.serialize` 는 torch 만 있으면 동작하는데, 여기에
ultralytics import 를 넣으면 직렬화 규약만 쓰려는 쪽까지 무거운 의존성을 끌게 된다.
"""
