'''
・Envの調整
・DQNを導入
・行動と状態をDQNと合併
'''

import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
import time, json, torch, os
import matplotlib.pyplot as plt
from dqn_lib import N_Network, ReplayBuffer
from wrapper_gpt import TwoWheeledInvertedPendulumEnv

# Mujocoの初期設定
model = mujoco.MjModel.from_xml_path('two_wheel_robot/scene.xml')
data = mujoco.MjData(model)

# ハイパーパラメータを取得
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

# gymの環境設定
# env=gym.make('CartPole-v1')
env = TwoWheeledInvertedPendulumEnv(model, data)
render_mode = "cpu"
env.render(render_mode=render_mode)
# env=gym.make('CartPole-v1', render_mode='human')
total_reward = 0
# print(env.reset())
(state,_)=env.reset()
rotmat = data.xmat[1].reshape(3, 3)
rot = R.from_matrix(rotmat)
euler = rot.as_euler('xyz', degrees=True)
print(data.xpos, euler)
time_step = 0
average = 20
reward_datas = []
MODEL_SAVE_PATH = "mujoco_Cpole_weights.pth"  # モデルの重みを保存するパス
Qnet = N_Network()
Qnet.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu")) # 学習済みモデルの重みを読み込む
Qnet.eval()
        

def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0) # PyTorchのTensorに変換＋バッチ形式に対応
    with torch.no_grad(): # 勾配計算を無効化：順伝搬してるだけで学習はしなくていいから
        # print(Qnet(state), Qnet(state).argmax())
        return Qnet(state).argmax().item()

state, _ = env.reset()
total_reward = 0
time_step = 0 # 本質はtotal_rewardと同じだが分かり易くするために分けた
episode_over = False

# ビューワー起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_start = time.time()
    print("2秒後に開始します")
    time.sleep(2)

    while not episode_over:
        # 行動の実行
        action = select_action(state)  # 現在の状態から方策に従って行動を選択
        next_state, reward, terminated, truncated, data, info = env.step(action) # 選択された行動によって環境を更新
        episode_over = terminated or truncated

        state = next_state
        
        # print(next_state, reward, terminated, truncated, info)
        total_reward += reward
        time_step += 1
        # w = Qnet.state_dict()

        viewer.sync()
        rotmat = data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        euler = rot.as_euler('xyz', degrees=True)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # print(model.opt.timestep, time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# imageio.mimsave("cartpole.gif", frames, fps=30)
env.close()