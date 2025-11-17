# ClimateToText
## Architecture
<img width="1471" height="1029" alt="image" src="https://github.com/user-attachments/assets/8d9662f5-6253-40f1-aec7-2747c756d846" />

## Code Structure
### `script/`
* 실행 가능한 개별 main 함수를 포함한 각 step 별 training, inference code
* `train_image_encoder.py` : Perceiver encoder 기반의 stage 1 이미지 인코더 학습 코드
### `src/`
* `script/` 폴더에 구현된 각 step에 필요한 파이썬 모듈 정의
* `data.py` : `Dataset` 과 `DataLoader`에 필요한 클래스 정의
* `losses.py` : InfoNCE와 같은 loss function utility 함수 정의
* `models.py` : 인코더, attention block과 같은 각종 neural network 구성 요소 정의
