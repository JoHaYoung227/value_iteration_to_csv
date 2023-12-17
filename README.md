# value_iteration_to_csv

## 환경 변수 설정 (environment.py)

`environment.py` 파일에서는 강화 학습 환경의 변수를 설정합니다. 
해당 파일은 주어진 CSV 파일을 읽어와서 강화 학습에 필요한 상태, 행동, 보상을 정의합니다.


## value iteration 알고리즘 실행 (value_iteration.py)
`value_iteration.py` 파일에서는 Dynamic Programming의 value iteration 알고리즘을 실행합니다.

## 데이터 파일 경로 설정
변수 설정 시 주의사항:
- `data_path`: CSV 파일의 경로를 지정합니다. 각 코드 파일에서 이 변수를 적절한 파일 경로로 변경해야 합니다.
- 각 코드 파일에서 `data_path` 변수를 CSV 파일의 경로로 수정합니다. 예를 들어:
```python
data_path = "your/path/to/data.csv"


