# environment.py
import pandas as pd

class Environment:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        # Add other initialization logic as needed

    def get_states(self):
        # Define how to extract states from the dataframe
        states = self.df[['기상상태', '사고유형', '노면상태', '가해운전자 차종', '가해운전자 상해정도', '피해운전자 차종', '시간', '도로형태1', '도로형태2']].drop_duplicates().apply(tuple, axis=1).tolist()
        return states

    def get_actions(self):
        # Define how to extract actions from the dataframe
        actions = self.df['법규위반'].unique().tolist()
        return actions

    def get_rewards(self, state, action):
        # Define how to calculate rewards based on state and action
        state_action_rewards = self.df[(self.df['기상상태'] == state[0]) & (self.df['사고유형'] == state[1]) & (self.df['노면상태'] == state[2]) & (self.df['가해운전자 차종'] == state[3]) & 
                                  (self.df['가해운전자 상해정도'] == state[4]) & (self.df['피해운전자 차종'] == state[5]) & (self.df['시간'] == state[6]) & 
                                  (self.df['도로형태1'] == state[7]) & (self.df['도로형태2'] == state[8]) & (self.df['법규위반'] == action)]['ECLO']
        return state_action_rewards.mean() if not state_action_rewards.empty else 0

if __name__ == "__main__":
    # 환경 클래스를 사용하여 데이터를 읽고 상태 및 행동을 확인하는 코드
    data_path = "example.csv"
    env = Environment(data_path)
    states = env.get_states()
    actions = env.get_actions()
    print(f"States: {states}")
    print(f"Actions: {actions}")
