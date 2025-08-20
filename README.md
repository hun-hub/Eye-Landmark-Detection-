# 👁 Eye Landmark Detection  
### "Optimized Stacked Hourglass Networks for Efficient Eye Landmark Detection," Engineering Application of Artificial Intelligence


### 눈 단일 영역 이미지를 대상으로 Eye keypoints (동공, 홍채 등) 탐지하는 모델 
![Eye keypoints](https://github.com/hun-hub/Eye-Landmark-Detection-/blob/master/Keypoints%20Detection.png)

## 📖 Description 
선행 연구에서는 얼굴 전체 영역을 입력으로 받아 눈 영역을 선행적으로 bounding box와 같은 것으로 detection 한 후 Eye keypoints들을 detection 하는 방식이었다. 
우리 연구의 경우 눈 단일 이미지만을 입력으로 받아 Eye keypoints들을 detection 하는 방향으로 모델 설계 및 구축하였다. 
이는, 최근 비전프로와 같이 눈에 착용하는 기기의 활용도가 올라감에 따라, 이 모델이 메타버스와 같은 분야에서의 발전에 기여할 것을 기대한다.  


## 📂 Data set 
1. IREYE4TASK (적외선 눈 영상 => 이미지로 가공)
2. UnityEyes – a tool for rendering eye images ( 눈 각도와 같은 Parameter 조정하면서 rgb 눈 이미지 generating)

## 💻 Run a Program
- **Train** 

학습을 진행하고 싶으면, 다음과 같은 폴더 경로를 따라, 상황에 맞게 argument를 조정하며 학습을 진행하면 된다.
폴더 경로: Eye-Landmark-Detection-main / train_mpii.py

- **Program operation**

눈 영역 이미지나 영상에 대해 checkpoint_20.pth.tar의 가중치를 사용해 Eye keypoints detection을 실행하면 된다. 
Eye-Landmark-Detection-main/test_metric.ipynb 에 예시로 구현하였다. 

## 👨‍👨‍👦 Developer & Contribution

**Developer** 

- Suk Hun Ko (me)

- Seung Gun Lee

- Yeong Je Park



**Contribution (me)** 


- 적외선 눈 영상 프레임 단위로 이미지 변환 및 Generator를 통한 rgb 눈 이미지 데이터 셋 구축 

- 이미지 특성을 고려한 loss function 설계 ( Adaptive Wing loss, MSE ...etc.)

- task에 적절한 base model 선정 및 stacked hourglass networks (base model) 최적의 stack 도출 






