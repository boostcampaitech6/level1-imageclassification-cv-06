# [네이버 부스트캠프 AI Tech 6기] Mask Classification Competition

## 프로젝트 소개

- Input data : 2,700명의 동양인 이미지. 한 명당 7장 _(마스크 착용*5, 미착용*1, 이상 착용1)_
- Output : test 이미지에 대한 분류 값 (18개 클래스)
  
#### Output class
3 * 2 * 3 = 18 class
- Mask : (Wear, Incorrect, Not Wear)
- Gender : (Male, Female)
- Age : (<30, >=30 and <60, >=60)
  

  
#### 평가방법
- F1 Score

<br>

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


<br>


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

Task 별 best_model 적용 방법

1. 가장 좋은 것 같은 모델을 선택한다.
2. 해당 모델의 저장된 학습 결과 {task}_best_model.pth 파일을 best_model/ 하위로 옮겨준다. 이때 덮어쓰기 여부를 묻는다면 덮어쓴다.
3. models/{task}_model.py 파일에 자신이 (2)에서 사용했던 모델의 Class가 정의되어 있는지 확인한다. (이 부분을 수행하지 않으면 추후 Inference 시 문제가 발생)
4. inference.sh 파일을 실행하기 전, argument를 세팅한다.


예시
```
python inference.py \

(추론 데이터셋 경로, eval 폴더를 기준으로 설정)
--data_dir /data/ephemeral/home/data/eval \

(best_model 폴더 경로, 프로젝트 내 best_model 폴더를 기준으로 설정)
--model_dir /data/ephemeral/home/project/repo/level1-imageclassification-cv-06/best_model \

(submission 파일이 나오는 최종 경로)
--output_dir ./ \ 

(models/age_model.py 에 정의되어 있는 Class 이름. 반드시 best_model/age_best_model.pth와 동일한 모델이 선택되어야함)
--age_model MyModel \ 

(models/gender_model.py 에 정의되어 있는 Class 이름. 반드시 best_model/gender_best_model.pth와 동일한 모델이 선택되어야함)
--gender_model MyModel \

(models/mask_model.py 에 정의되어 있는 Class 이름. 반드시 best_model/mask_best_model.pth와 동일한 모델이 선택되어야함)
--mask_model MyModel \

(각 모델별 사용하는 Data augmentation을 설정. 설정된 Augmentation이 dataset.py에 반드시 정의되어 있어야 함)
--age_augmentation BaseAugmentation \
--gender_augmentation BaseAugmentation \
--mask_augmentation BaseAugmentation \
```
