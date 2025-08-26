# [식음업장 메뉴 수요 예측 대회](https://dacon.io/competitions/official/236559/overview/description)

### 배경
 리조트 내 식음업장에서의 실전 수요 예측 문제를 AI로 해결해보는 것을 목표로 합니다. Aimers 여러분들은 실제 식음업장에서 수집된 판매 데이터를 기반으로, 각 메뉴가 1주일 동안 얼마나 판매될지를 예측하는 모델을 개발

### 데이터
```
├── 📁 train/
│   └── train.csv
├── 📁 test/
│   ├── TEST_00.csv
│   ├── TEST_01.csv
│   └── ... (총 10개 파일)
└── sample_submission.csv
```

1. `train.csv`
- 모델 학습을 위한 과거 매출 기록 데이터
- 기간 : 2023년 1월 1일 ~ 2024년 6월 15일
- 세부내용
    - 영업일자 : 매출이 발생한 날짜(YYYY-MM-DD)
    - 영업장명_메뉴명 : 각 판매 아이템의 고유 식별자
    - 매출수량 : 해당 아이템의 일일 판매량

2. `TEST_XX.csv`
- 예측을 수행해야 할 대상 기간의 데이터
- 기간 : `TEST_XX.csv` 파일의 마지막 날짜로부터 **미래 7일간**의 매출 수량을 예측하여 제출

### 평가
대회의 평가 지표는 각 식음업장($s$)의 개별 품목($i$)에 대한 SMAPE(Symmetric Mean Absolute Percentage Error)를 계산한 후, 식음업장의 가중치($w_s$)를 적용하여 최종 점수를 산출합니다.

$$\text{Score} = \sum_s w_s \cdot \left( \frac{1}{|I_s|} \sum_{i \in I_s} \left( \frac{1}{T_i} \sum_{t=1}^{T_i} \frac{2|A_{t,i} - P_{t,i}|}{|A_{t,i}| + |P_{t,i}|} \right) \right)$$

-   $s$ : 식음업장명
-   $w_s$ : 식음업장 $s$의 가중치 (비공개)
-   $I_s$ : 식음업장 $s$에 속한 품목 컬럼 집합
-   $T_i$ : 품목 $i$에서 유효한 날짜 수 ($A_{t,i} \neq 0$)
-   $A_{t,i}, P_{t,i}$ : 날짜 $t$, 품목 $i$의 실제값과 예측값

### 일정
- 2025년 08월 01일(금) 10:00 ~ 2025년 08월 25일(월) 10:00

### 결과
- Public  : [**39** / 820]
- Private : [**55** / 820]