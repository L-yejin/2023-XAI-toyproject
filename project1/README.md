# 인공지능 기반 상품 및 인물 사진 컨셉 합성 솔루션

#### 사용 모델: [segmentation_models_pytorc](https://github.com/qubvel/segmentation_models.pytorch)
사용시 해당 GitHub clone 해서 사용해야합니다

## 1. 배경 및 목적
- 제품 사진 촬영 시 발생되는 비용과 인력 문제에 대한 소상공인의 부담 증가
- 제품과 인물 사진에 인공지능을 활용하여 배경 이미지와 사진을 자동 합성하는 솔루션 제작

## 2. 주최/주관 &팀원
- [2023년 데이터바우처 지원사업]
- 팀원: 총 4명

## 3. 프로젝트 기간
- 2023.06.13 ~ 2023.08.10

## 4. 프로젝트 소개
<img width="762" alt="스크린샷 2023-08-21 오후 3 34 32" src="https://github.com/L-yejin/2023-XAI-toyproject/assets/104400282/247b959f-606c-4fe6-ba85-57ce8b13ba28">

  합성을 원하는 객체 이미지와 배경을 선택하면 **이미지를 자동으로 합성**하는 솔루션을 만드는 프로젝트이다. 10가지 객체와 배경 100장을 합성하여 총 1,000장의 결과물과 시현을 할 수 있어야 했다.

  객체 10개에 대해서 **segmentation_models_pytorch**를 사용하여 학습을 진행하였다. 학습된 모델을 통해 **객체 segmentation**을 진행하였고, 이를 통해 나온 mask를 사용하여 **객체만 추출**하는 작업을 거쳤다. 이렇게 추출된 **객체와 원하는 배경에 합성**하였다. 

  시현이 가능하도록 Python에서 가능한 GUI 개발 라이브러리 중 Qt 기반의 **PySide6**를 사용하여 구현하였다.
![녹화본](https://github.com/L-yejin/2023-XAI-toyproject/assets/104400282/edd65fb8-ef42-4eb6-a8df-264b89d00a5f)

### 관련 자료
[X:AI 중간 발표 자료](https://drive.google.com/file/d/1H5PmitwSkeTHeQti3sVBzat_c2JHoJRe/view?usp=drive_link)
