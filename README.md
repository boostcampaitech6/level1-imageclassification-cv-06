# CV Level1 - Image Classification

## 프로젝트 구조

```
${project}
├── dataset.py
├── inference.py
├── loss.py
├── model
│   ├── age_model.py
│   ├── gender_model.py
│   └── mask_model.py
├── train.py
├── result
│   ├── age.csv
│   ├── gender.csv
│   ├── mask.csv
│   └── submission.csv
└── requirements.txt
```

- dataset.py : Pytorch의 Dataset을 구현한 클래스. 모든 Model에서 공통으로 참조한다.
- inference.py : Eval Dataset에 대하여 결과를 출력한다.
- loss.py : Loss Function이 구현되어 있는 파일.
- model/* : 세 가지 분류에 대한 모델이 들어있다.
- train.py : 각 모델에 대한 training을 담당
- result : inference에 대한 결과를 저장한다.
- README.md
- requirements.txt : contains the necessary packages to be installed

## 사용방법

### venv 설정

1. 프로젝트를 위한 가상환경을 세팅합니다.

```
# 프로젝트 directory로 이동
cd ${project}
# 파이썬 가상환경 만들기
python -m venv {가상환경 path}
```

3. 가상환경을 활성화 합니다.

```
source {가상환경 path}/bin/activate
```

4. 가상환경을 비활성화 하려면..

```
deactivate
```

### 요구 패키지 설치

`requirements.txt`에 명시되어 있는 패키지를 설치합니다.

```
# (*주의)가상환경을 Activate 한 후에 실행시켜야 합니다.
pip install -r requirements.txt
```

### 사용법

#### 학습

아래의 두 명령어 중 하나를 이용해 모델을 학습시킵니다.

```
SM_CHANNEL_TRAIN=/path/to/images SM_MODEL_DIR=/path/to/model python train.py
```

```
python train.py --data_dir /path/to/images --model_dir /path/to/model
```

#### 추론

아래의 두 명령어 증 하나를 통해 기학습된 모델로 특정 데이터셋을 추론합니다.

```
SM_CHANNEL_EVAL=/path/to/images SM_CHANNEL_MODEL=/path/to/model SM_OUTPUT_DATA_DIR=/path/to/output python inference.py
```

```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model
```