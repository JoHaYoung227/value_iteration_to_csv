import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# train.csv 파일 로드
df = pd.read_csv("C:/Users/johy3/OneDrive/바탕 화면/value_iteration_to_csv-main/value_iteration_to_csv-main/example.csv")

# 상태와 행동 정의
states = df[['기상상태', '사고유형', '노면상태', '가해운전자 차종', '가해운전자 상해정도', '피해운전자 차종', '시간', '도로형태1', '도로형태2']].drop_duplicates().apply(tuple, axis=1).tolist()
actions = df['법규위반'].unique().tolist()

# 상태-행동 쌍에 대한 보상 함수 초기화
rewards = {}
for state in states:
    for action in actions:
        # 해당 상태와 행동에서의 평균 보상 계산
        state_action_rewards = df[(df['기상상태'] == state[0]) & (df['사고유형'] == state[1]) & (df['노면상태'] == state[2]) & (df['가해운전자 차종'] == state[3]) &
                                  (df['가해운전자 상해정도'] == state[4]) & (df['피해운전자 차종'] == state[5]) & (df['시간'] == state[6]) &
                                  (df['도로형태1'] == state[7]) & (df['도로형태2'] == state[8]) & (df['법규위반'] == action)]['ECLO']
        rewards[(state, action)] = state_action_rewards.mean() if not state_action_rewards.empty else 0

# 상태에 대한 가치 함수 초기화
values = {state: 0 for state in states}

# Value Iteration 알고리즘
gamma = 0.9  # 할인 계수
threshold = 1  # 수렴 기준
iterations = 0
delta_values = []  # max delta 값을 저장할 리스트 추가

while True:
    delta = 0
    for state in states:
        v = values[state]
        values[state] = max([rewards[(state, action)] + gamma * max(values[s] for s in states if s != state) for action in actions])
        delta = max(delta, abs(v - values[state]))
    iterations += 1
    delta_values.append(delta)  # max delta 값을 리스트에 추가
    print(f"Iteration: {iterations}, Max Delta: {delta}")
    print(f"Current Value Function: {values}")
    if delta < threshold:
        break

# 최적의 결정론적 정책 추출
optimal_policy = {}
for state in states:
    best_action = max(actions, key=lambda action: rewards[(state, action)] + gamma * values[state])
    optimal_policy[state] = best_action

# 결과 출력
print("Optimal Policy:")
for state, action in optimal_policy.items():
    print(f"State: {state}, Optimal Action: {action}")

# 그래프 그리기
plt.plot(range(1, iterations + 1), delta_values, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Max Delta')
plt.title('Convergence of Value Iteration')
plt.show()

# 최적 정책을 DataFrame으로 변환
optimal_policy_df = pd.DataFrame(list(optimal_policy.items()), columns=['State', 'Optimal Action'])

# 테이블 형태로 출력
print("Optimal Policy:")
print(optimal_policy_df)

# DataFrame을 CSV 파일로 저장
optimal_policy_df.to_csv('optimal.csv', index=False)