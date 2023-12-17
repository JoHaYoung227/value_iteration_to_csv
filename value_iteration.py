# value_iteration.py
import matplotlib.pyplot as plt

# 가치 반복(Value Iteration) 알고리즘을 구현한 클래스 정의
class ValueIteration:
    def __init__(self, environment, gamma=0.9, threshold=0.01):
        # 환경 객체 및 알고리즘 매개변수 초기화
        self.environment = environment
        self.gamma = gamma
        self.threshold = threshold

    # 가치 반복 알고리즘 실행 메서드
    def run(self):
        # 환경으로부터 상태와 행동 공간 추출
        states = self.environment.get_states()
        actions = self.environment.get_actions()

        # 각 상태의 초기 가치를 0으로 설정
        values = {state: 0 for state in states}
        iterations = 0
        delta_values = []  # max delta 값을 저장할 리스트 추가

        while True:
            delta = 0
            # 모든 상태에 대해 가치 업데이트 수행
            for state in states:
                v = values[state]
                # 벨만 최적 방정식을 이용한 가치 업데이트
                values[state] = max([self.environment.get_rewards(state, action) + self.gamma * max(values[s] for s in states if s != state) for action in actions])
                # 변화의 최대값 갱신
                delta = max(delta, abs(v - values[state]))
            iterations += 1
            delta_values.append(delta)  # max delta 값을 리스트에 추가
            print(f"Iteration: {iterations}, Max Delta: {delta}")
            print(f"Current Value Function: {values}")

            # 변화가 일정 임계값 이하면 알고리즘 종료
            if delta < self.threshold:
                break

            # 최적의 결정론적 정책 추출
            optimal_policy = {}
            for state in states:
                best_action = max(actions, key=lambda action: self.environment.get_rewards(state, action) + self.gamma * values[state])
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

if __name__ == "__main__":
    환경과 가치 반복 클래스를 사용하여 알고리즘 실행하는 코드
    data_path = "example.csv" # 여기에서 데이터 경로를 바꾸세요.
    from environment import Environment
    env = Environment(data_path)
    value_iteration = ValueIteration(env)
    value_iteration.run()
