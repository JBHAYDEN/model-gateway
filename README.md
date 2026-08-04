# Model Gateway

고객 전환 예측을 위한 머신러닝 학습 파이프라인과 FastAPI 추론 API입니다.
합성 데이터 생성, 특징 공학, 전처리, 교차 검증, 성능 평가, 모델 저장 및 API 서빙을 포함합니다.

## Tech Stack
- Python 3.12
- pandas / NumPy
- scikit-learn
- Pipeline / ColumnTransformer
- StandardScaler / OneHotEncoder
- LogisticRegression / GridSearchCV
- Accuracy / Precision / Recall / F1-score
- joblib
- FastAPI / Pydantic
- Docker
- pytest / TestClient

## Train
```bash
pip install -r requirements.txt
python scripts/train_model.py
```

## Run API
```bash
uvicorn app.main:app --reload
```

## Endpoints
- `GET /health`
- `GET /model-info`
- `POST /predict`

```json
{
  "age": 31,
  "sessions": 8,
  "purchases": 2,
  "averageSessionMinutes": 12.5,
  "channel": "organic"
}
```

This repository is a small educational machine-learning project.
