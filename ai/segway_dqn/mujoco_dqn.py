'''
そろそろパラメータと結果の対応を記録しよう。
その前に初期ノイズ入れるか

1201_2:学習率低減、ε減衰をより緩やかに
1201_2:行動サイズを11から1023に変更、初期epsilonを0.8から0.6に
1204_2:報酬にトルク含む
'''

import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
import time, json, torch, os
import matplotlib.pyplot as plt
from dqn_lib import N_Network, PrioritizedReplayBuffer  # PERバッファをインポート
from wrapper_gpt import TwoWheeledInvertedPendulumEnv
import torch.nn.functional as F

# Mujocoの設定
model = mujoco.MjModel.from_xml_path('two_wheel_robot/scene.xml')
data = mujoco.MjData(model)

# パラメータ読み込み（dqn_libと同様）
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "parameters_dqn.json")
with open(json_path, 'r') as f:
    json_param = json.load(f)["dqn_cartpole"]
parameters = json_param # 簡略化のためそのまま代入

# 環境とネットワーク設定
env = TwoWheeledInvertedPendulumEnv(model, data)
render_mode = "cpu"
preliminary_total_reward = 5000

Qnet = N_Network()
Tnet = N_Network()
Tnet.load_state_dict(Qnet.state_dict())
Tnet.eval()

# PER用バッファに変更
Replay_Buffer = PrioritizedReplayBuffer(parameters["buffer_size"], parameters["batch_size"])

# ★重要修正: オプティマイザはループの外で一度だけ定義
var_lr = parameters["lr"]
optimizer = torch.optim.Adam(Qnet.parameters(), lr=var_lr)

change_epsilon = parameters["epsilon"]
global_step = 0 # ターゲットネットワーク更新用（エピソードまたぎ）
td_interval = parameters["td_interval"]
reward_datas = []

MODEL_SAVE_PATH = "mujoco_Cpole_weights_1204_2.pth"  # モデルの重みを保存するパス


def select_action(state):
    global change_epsilon
    if np.random.rand() <= change_epsilon:
        return np.random.randint(0, parameters["action_size"])
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return Qnet(state).argmax().item()

def update_model():
    """TD誤差計算と重み更新"""
    # PERサンプリング (インデックスとIS重みも受け取る)
    states, actions, rewards, next_states, dones, idxs, is_weights = Replay_Buffer.sampling()

    actions = actions.unsqueeze(1)
    q_values = Qnet(states).gather(1, actions).squeeze()

    with torch.no_grad():
        next_q_max = Tnet(next_states).max(1)[0]
        q_value_target = rewards + (1 - dones) * parameters["gamma"] * next_q_max

    # PER用に個別のTD誤差（絶対値）を計算（勾配計算は不要なのでno_grad推奨だが、ここではLoss計算に使うため保持）
    td_errors = torch.abs(q_value_target - q_values).detach().numpy()

    # 優先度の更新
    Replay_Buffer.update_priorities(idxs, td_errors)

    # 損失関数の計算（IS重みを適用）
    # 重み付きMSE: (w * (Q - Target)^2).mean()
    loss = (is_weights * F.mse_loss(q_values, q_value_target, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

diff_epsilon = 1e-5
def train_step(state, episode):
    global global_step, change_epsilon, optimizer, var_lr

    action = select_action(state)
    next_state, reward, terminated, truncated, data, info = env.step(action)
    episode_over = terminated or truncated

    Replay_Buffer.add(state, action, reward, next_state, episode_over)

    # バッチサイズ以上溜まったら学習
    if len(Replay_Buffer) >= parameters["batch_size"]:
        update_model()

    # ターゲットネットワークの同期 (global_stepで管理)
    if global_step % parameters["td_interval"] == 0:
        Tnet.load_state_dict(Qnet.state_dict())
        print("Target Network Updated")
    
    # εの減衰 (ステップごと or エピソードごと。ここではステップごとに少しずつ)
    if change_epsilon > 0.02:
        diff_epsilon = (parameters["episodes"] - episode) / 300 * 1e-6
        if change_epsilon < 0.1:
            var_lr -= (parameters["episodes"] - episode) / 6000 * 1e-9
            var_lr = max(var_lr, 0.00001)
            # ★修正2: 計算した var_lr をオプティマイザに反映させる！
            for param_group in optimizer.param_groups:
                param_group['lr'] = var_lr
            diff_epsilon = (parameters["episodes"] - episode) / 2500 * 1e-6
        elif change_epsilon < 0.2:
            diff_epsilon = (parameters["episodes"] - episode) / 2000 * 1e-6
        elif change_epsilon < 0.3:
            diff_epsilon = (parameters["episodes"] - episode) / 1500 * 1e-6
        elif change_epsilon < 0.5:
            diff_epsilon = (parameters["episodes"] - episode) / 1000 * 1e-6
            
        change_epsilon -= diff_epsilon

    global_step += 1
    return next_state, reward, episode_over

for episode in range(parameters["episodes"]):
    state, _ = env.reset()
    time_step = 0
    total_reward = 0
    episode_over = False
    
    # 描画モードの判定（humanかつ30エピソードごとの場合のみ描画）
    is_render_episode = (render_mode == "human" and episode % 50 == 0)

    if is_render_episode:
        # 描画ありの実行
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"Episode {episode}: Rendering...")
            step_start = time.time()
            
            while not episode_over:
                # 学習ステップの実行
                state, reward, episode_over = train_step(state, episode)
                
                total_reward += reward
                time_step += 1
                
                # ビューワーの同期
                viewer.sync()
                
                # 実時間同期（シミュレーション速度を現実に合わせる処理）
                time.sleep(model.opt.timestep)
                # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)
                # step_start = time.time()
    else:
        # 描画なしの実行（高速）
        while not episode_over:
            state, reward, episode_over = train_step(state, episode)
            total_reward += reward
            time_step += 1

    # if(time_step < parameters["max_step"] / 1.5):
    #     total_reward += time_step / 5  # 最大ステップ到達ボーナス
        # print(time_step / 2)

    # エピソード終了後の処理
    reward_datas.append(total_reward)
    
    # 成功判定ロジック（元のコードを踏襲）
    # if total_reward >= 2000 and var_lr > 0.0001:
    #     var_lr *= 0.9
    #     print(var_lr)
            
    if (total_reward >= preliminary_total_reward):
        try:
            preliminary_total_reward = total_reward
            torch.save(Qnet.state_dict(), MODEL_SAVE_PATH)
            print(f"モデルの重みを {MODEL_SAVE_PATH} に正常に保存しました。")
        except Exception as e:
            print(f"モデルの重みの保存中にエラーが発生しました: {e}")
    
    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Time Step: {time_step}, Epsilon: {change_epsilon:.4f}")

# try:
#     torch.save(Qnet.state_dict(), MODEL_SAVE_PATH)
#     print(f"モデルの重みを {MODEL_SAVE_PATH} に正常に保存しました。")
# except Exception as e:
#     print(f"モデルの重みの保存中にエラーが発生しました: {e}")