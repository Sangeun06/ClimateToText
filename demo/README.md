# 기상 AI 서비스

FastAPI 기반의 간단한 기상 관련 AI 서비스입니다.

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
conda activate base && python main.py
CUDA_VISIBLE_DEVICES=1 conda activate qwenfin && python -m sglang.launch_server --model-path ~/yjju/EarthData/output/qwen3_weather_merged --port 30000 --host 0.0.0.0
conda activate base && python modis_server_standalone.py
```

- `qwen-agent`가 설치되어 있으면 자동으로 MCP 서버를 띄워 툴을 사용합니다. 로컬 OpenAI 호환 엔드포인트(`http://localhost:30000/v1`)와 함께 동작하도록 설정되어 있습니다.

또는

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 접속

브라우저에서 http://localhost:8000 으로 접속하세요.

## API 엔드포인트

- `GET /` - 웹 인터페이스
- `POST /predict` - 질문 제출 (폼)
- `GET /api/predict?query=질문` - JSON API
- `GET /health` - 헬스 체크

## 예시 질문

- "오늘 날씨는 어떤가요?"
- "비가 올 확률은 얼마나 되나요?"
- "현재 기온은 몇 도인가요?"
- "바람은 얼마나 세게 부나요?"
- "습도가 높은 이유는 무엇인가요?"
