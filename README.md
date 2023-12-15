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
# 1. 파이썬 가상환경 만들기
python -m venv {가상환경 path}

# 2. 파이썬 가상환경 활성화
source {가상환경 path}/bin/activate

# 3. 필요 패키지 설치
pip install -r requirements.txt

# 4. 작업이 끝난 후 가상환경 비활성화
deactivate
```

### 사용법

#### 학습

1. 쉘 스크립트 내용을 알맞게 수정 
```
(train_age.sh 예시)
python train.py \
--data_dir /data/ephemeral/home/data/train/images \  (학습 데이터 path)
--model_dir /data/ephemeral/home/project/repo/level1-imageclassification-cv-06/ \  (모델 path)
--model_type age_model \ (어떤 모델을 사용할 것인지)
--dataset AgeModelDataset \
--criterion focal
```
2. 스크립트 실행
```
cd ${project}
./train_${task}.sh
```
#### 추론

(아직 추론 기능은 논의가 필요하여 나중에 개발 예정입니다!!)
