# value_iteration.py
from environment import Environment

class ValueIteration:
    def __init__(self, environment, gamma=0.9, threshold=0.01):
        self.environment = environment
        self.gamma = gamma
        self.threshold = threshold

    def run(self):
        states = self.environment.get_states()
        actions = self.environment.get_actions()

        values = {state: 0 for state in states}
        iterations = 0

        while True:
            delta = 0
            for state in states:
                v = values[state]
                values[state] = max([self.environment.get_rewards(state, action) + self.gamma * max(values[s] for s in states if s != state) for action in actions])
                delta = max(delta, abs(v - values[state]))
            iterations += 1
            print(f"Iteration: {iterations}, Max Delta: {delta}")
            print(f"Current Value Function: {values}")
            if delta < self.threshold:
                break

if __name__ == "__main__":
    # 예시: 환경과 가치 반복 클래스를 사용하여 알고리즘 실행하는 코드
    data_path = "example.csv"
    env = Environment(data_path)
    value_iteration = ValueIteration(env)
    value_iteration.run()