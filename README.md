# 👁 Eye Landmark Detection  



### 눈 단일 영역 이미지를 대상으로 Eye keypoints (동공, 홍채 등) 탐지하는 모델 
![Eye keypoints](https://github.com/hun-hub/Eye-Landmark-Detection-/blob/master/Keypoints%20Detection.png)

## 📖 Description 
선행 연구에서는 얼굴 전체 영역을 입력으로 받아 눈 영역을 선행적으로 bounding box와 같은 것으로 detection 한 후 Eye keypoints들을 detection 하는 방식이었다. 
우리 연구의 경우 눈 단일 이미지만을 입력으로 받아 Eye keypoints들을 detection 하는 방향으로 모델 설계 및 구축하였다. 
이는, 최근 비전프로와 같이 눈에 착용하는 기기의 활용도가 올라감에 따라, 이 모델이 메타버스와 같은 분야에서의 발전에 기여할 것을 기대한다.  


## 📂 Data set 
1. IREYE4TASK (적외선 눈 영상 => 이미지로 가공)
2. UnityEyes – a tool for rendering eye images ( 눈 각도와 같은 Parameter 조정하면서 rgb 눈 이미지 generating)

## 
