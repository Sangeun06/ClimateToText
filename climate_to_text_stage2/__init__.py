"""
Stage2: 이미지 → 텍스트 설명 생성 모듈 패키지.

이 패키지는 다음과 같은 구성 요소를 포함한다.

- dataset.py      : JSON 포맷의 (image, annotation, cond_name) 샘플을 PyTorch Dataset으로 변환
- stage2_model.py : 이미지 인코더 스텁 + LLM(Mistral)과의 접속부, prefix embedding 생성
- train_stage2.py : Stage2 파인튜닝 스크립트 (LoRA 기반)
"""

