"""통합형 VLM 학습·추론.

**이 파일은 영구히 비워 둔다.** `vlm/coords.py`는 채점 쪽이 import하는 의존성 제로 리프
모듈인데, 여기에 torch·transformers 같은 것을 import하면 `from vlm.coords import ...`
한 줄에 그 의존성이 딸려 들어가 리프 원칙이 깨진다. 패키지 초기화 코드가 필요하면
하위 모듈에 둔다. tests/test_coords.py가 이 파일이 비어 있는지 검사한다.
"""
