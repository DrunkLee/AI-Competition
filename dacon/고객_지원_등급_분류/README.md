# [데이콘 Basic 고객 지원 등급 분류](https://dacon.io/competitions/official/236562/overview/description)

### 배경
고객의 이용 행동과 계약 정보 등을 분석하여 어떤 고객이 더 많은 지원을 필요로 하는지 미리 예측할 수 있다면,
서비스 제공자는 보다 효율적으로 자원을 배분하고, 맞춤형 고객 응대를 실현할 수 있게 됩니다.
고객의 서비스 이용 데이터를 기반으로 지원 필요 수준을 분류하는 AI 알고리즘 개발

### 데이터
```
train.csv
test.csv
sample_submission.csv
```

1. `train.csv`
- ID : 샘플별 고유 ID
- age : 고객 나이
- gender : 고객 성별
- tenure : 고객이 서비스를 이용한 총 기간 (월)
- frequent : 고객의 서비스 이용일
- payment_interval : 고객의 결제 지연일
- subscription_type : 고객의 서비스 등급
- contract_length : 고객의 서비스 계약 기간
- after_interaction : 고객이 최근 서비스 이용으로부터 경과한 기간 (일)
- support_needs : 고객의 지원 필요도 (0 : 낮음 , 1 : 중간 , 2 : 높음)

2. `TEST_XX.csv`
- ID : 샘플별 고유 ID
- age : 고객 나이
- gender : 고객 성별
- tenure : 고객이 서비스를 이용한 총 기간 (월)
- frequent : 고객의 서비스 이용일
- payment_interval : 고객의 결제 지연일
- subscription_type : 고객의 서비스 등급
- contract_length : 고객의 서비스 계약 기간
- after_interaction : 고객이 최근 서비스 이용으로부터 경과한 기간 (일)

### 평가
- Macro F1-Score
- Public Score: 전체 테스트 데이터 중 사전 샘플링된 30%
- Private Score: 전체 테스트 데이터 100%

### 일정
- 2025년 08월 04일(월) 10:00 ~ 2025년 09월 30일(화) 10:00

### 결과
- Public  : [?? / ??]
- Private : [?? / ??]