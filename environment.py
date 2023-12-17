# environment.py
import pandas as pd

# 환경 클래스 정의 
class Environment:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path) # csv 파일에서 데이터 불러오기 

    # 상태(states) 추출 메서드 정의 
    def get_states(self):
        # 데이터프레임에서 중복을 제거하고 튜플로 변환하여 상태 리스트 반환
        states = self.df[['기상상태', '사고유형', '노면상태', '가해운전자 차종', '가해운전자 상해정도', '피해운전자 차종', '시간', '도로형태1', '도로형태2']].drop_duplicates().apply(tuple, axis=1).tolist()
        return states

    # 행동(actions) 추출 메서드 정의
    def get_actions(self):
        # 데이터프레임에서 법규위반 컬럼의 고유값을 추출하여 행동 리스트 반환
        actions = self.df['법규위반'].unique().tolist()
        return actions

    # 보상(rewards) 계산 메서드 정의
    def get_rewards(self, state, action):
        # 주어진 상태와 행동에 대한 보상 계산
        state_action_rewards = self.df[(self.df['기상상태'] == state[0]) & (self.df['사고유형'] == state[1]) & (self.df['노면상태'] == state[2]) & (self.df['가해운전자 차종'] == state[3]) & 
                                  (self.df['가해운전자 상해정도'] == state[4]) & (self.df['피해운전자 차종'] == state[5]) & (self.df['시간'] == state[6]) & 
                                  (self.df['도로형태1'] == state[7]) & (self.df['도로형태2'] == state[8]) & (self.df['법규위반'] == action)]['ECLO']
        # 평균 보상 반환 (데이터가 비어있으면 0 반환)
        return state_action_rewards.mean() if not state_action_rewards.empty else 0

if __name__ == "__main__":
    # 환경 클래스를 사용하여 데이터를 읽고 상태 및 행동을 확인하는 코드
    data_path = "example.csv" # 여기서 데이터 경로를 바꾸세요.
    env = Environment(data_path) 
    states = env.get_states()
    actions = env.get_actions()
    print(f"States: {states}")
    print(f"Actions: {actions}")
