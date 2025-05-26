'''
observation: ([カートの位置，カートの速度，ポールの角度，ポールの角速度]，reward, 終了条件×２，{Info})
(array([ 0.017, -0.01 ,  0.002, -0.01 ], dtype=float32), {})

rewardは常に1．rewardの累計を大きくするのが目標
更新されたQnetworkが学習結果．これを用いて最適な行動が決定できるようになる．意味わかんね．
ミニバッチ学習はあくまで学習の効率化を図るのであって，行動選択には直接的に関係しない．
間接的にはQ値の更新が関係する．

！！！！parameters.jsonファイルはターミナルのディレクトリに依存する！

-DQNワークフロー
1.環境の初期化（例: CartPole-v1）
2.経験再生バッファ（Replay Buffer）の準備
3.Qネットワークの定義（
-----------------------------------↑ここまでが初期化

4.ε-greedy法で行動選択（たまにランダムな行動をする）
5.ニューラルネットで状態→Q値を出力）
6.経験をバッファに保存
7.一定ステップごとに学習実行
8.ターゲットネットワークの更新（soft/hard update）

報酬が十分になったら学習終了
'''
import gymnasium as gym
import numpy as np
import time, json, torch, os
import matplotlib.pyplot as plt
from dqn_lib import N_Network, ReplayBuffer
import imageio
# 環境設定
# env=gym.make('CartPole-v1', render_mode='rgb_array')
env=gym.make('CartPole-v1', render_mode='human')
print(env.action_space.n)
total_reward = 0
(state,_)=env.reset()
time_step = 0
average = 20
reward_datas = []
frames = []
MODEL_SAVE_PATH = "dqn_Cpole_weights2.pth"  # モデルの重みを保存するパス

# dqn_lib.py内での絶対パス取得
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "parameters_dqn.json")

with open(json_path, 'r') as f:
    json_param = json.load(f)["dqn_cartpole"]
parameters = {"lr": json_param["lr"],
            "gamma": json_param["gamma"],
            "episodes": json_param["episodes"],
            "epsilon": json_param["epsilon"],
            "buffer_size": json_param["buffer_size"],
            "batch_size": json_param["batch_size"],
            "td_interval": json_param["td_interval"],
            "input_size":json_param["input_size"],
            "action_size": json_param["action_size"],
            "max_step": json_param["max_step"]
}
Qnet = N_Network()
Qnet.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu")) # 学習済みモデルの重みを読み込む
Qnet.eval()
# print(torch.load(MODEL_SAVE_PATH))


# ε-greedyポリシーに従って行動を選択
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0) # PyTorchのTensorに変換＋バッチ形式に対応
    with torch.no_grad(): # 勾配計算を無効化：順伝搬してるだけで学習はしなくていいから
        # print(Qnet(state), Qnet(state).argmax())
        return Qnet(state).argmax().item()

# episodes回学習をする
# for episode in range(parameters["episodes"]):
    # 環境のリセット
state, _ = env.reset()
total_reward = 0
time_step = 0 # 本質はtotal_rewardと同じだが分かり易くするために分けた
episode_over = False

while not episode_over:
    # 行動の実行
    action = select_action(state)  # 現在の状態から方策に従って行動を選択
    next_state, reward, terminated, truncated, info = env.step(action) # 選択された行動によって環境を更新
    episode_over = terminated or truncated
    total_reward += 1
    time_step += 1
    # if np.random.rand() <= 0.1:
    #     print("do")
    #     next_state[1] += np.random.normal(0, 500) # カート速度に外乱
    #     next_state[3] += np.random.normal(0, 500) # ポール角速度に外乱
    frame = env.render()
    frames.append(frame)
    state = next_state

# 報酬データを保存
reward_datas.append(total_reward)
    # print(f"Episode: {episode}, Total Reward: {total_reward}, Time Step: {time_step}")
# GIF作成
# imageio.mimsave("cartpole.gif", frames, fps=30)
env.close()