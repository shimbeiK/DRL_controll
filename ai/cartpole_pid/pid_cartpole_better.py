'''
observation: ([カートの位置，カートの速度，ポールの角度，ポールの角速度]，reward, 終了条件×２，{Info})
(array([ 0.017, -0.01 ,  0.002, -0.01 ], dtype=float32), {})

https://zenn.dev/teruyamato0731/scraps/66339a544c0019
曰く，PID制御の目標は偏差Erを０に近づけること
en(t) = rn(t) - yn(t) #r:目標値, y:システムの状態の実値．目標値は位置と速度に重みをつけて計算する．
操作量 un(t) = Kp*en(t) + Kd*(en - en-1)/dt + Ki*Σeh*Δt(h=0➡n)
= Kp_position*en(t) + Kp_velocity*en(t) + Kd_position*vel + Kd_velocity*angler_vel➡０
操作量は離散化し，パラメータ調整をする
目標値が２変数で済んでいるから重みをつけて１変数にしているが４脚ロボットには適用不可．
'''
import gymnasium as gym
import numpy as np
import time, json, torch, os
import matplotlib.pyplot as plt
import imageio


# 環境設定
env=gym.make('CartPole-v1', render_mode='rgb_array')
# env=gym.make('CartPole-v1', render_mode='human')
total_reward = 0
(state,_)=env.reset()
time_step = 0
average = 20
episodes = 50
reward_datas = []

# dqn_lib.py内での絶対パス取得
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "cartpole_dqn/parameters_dqn.json")

Kp_position = 4
Kp_velocity = 20
Kd_position = 1
Kd_velocity = 5
Ki = 0
TARGET = 0 #nステップ目での目標値
actual = 0 #nステップ目での実値
output = 0
position_error = 0
velocity_error = 0
integral = 0
frames = []

# ε-greedyポリシーに従って行動を選択
def select_action(state):
    global integral, pre_error, errors
    # 偏差を計算
    time.sleep(0.05)
    # pd制御（偏差のずれがないので）
    actual = (Kp_position*state[0] + Kp_velocity*state[2]) + (Kd_position*state[1] + Kd_velocity*state[3])
    print(actual)
    error = TARGET - actual
    errors.append(error)

    # 操作量を計算
    output = - actual

    # 操作量を離散化して返す
    if output < 0:
        return 1
    else:
        return 0

# 環境のリセット
state, _ = env.reset()
total_reward = 0
time_step = 0 # 本質はtotal_rewardと同じだが分かり易くするために分けた
episode_over = False
linear_vels = []
angular_vels = []
errors = []

while not episode_over:
    print("in")
    # 行動の実行
    action = select_action(state)  # 現在の状態から方策に従って行動を選択
    next_state, reward, terminated, truncated, info = env.step(action) # 選択された行動によって環境を更新
    episode_over = terminated or truncated
    total_reward += 1
    time_step += 1

    # 外乱
    if np.random.rand() <= 0.1:
        print("do")
        next_state[1] += np.random.normal(0, 500) # カート速度に外乱
        next_state[3] += np.random.normal(0, 500) # ポール角速度に外乱
    frame = env.render()
    frames.append(frame)

    state = next_state
    linear_vels.append(state[1])
    angular_vels.append(state[3])

# 報酬データを保存
reward_datas.append(total_reward)
# print(f"Episode: {episode}, Total Reward: {total_reward}, Time Step: {time_step}")
# グラフの作成
# plt.figure(figsize=(12, 6))
# print(state[0], state[2])
# # 元の報酬データをプロット (任意)
# plt.plot(linear_vels, label='lonear_vel', alpha=0.8, color='blue')
# plt.plot(angular_vels, label='angular_vel', alpha=0.8, color='red')
# plt.plot(errors, label='error', alpha=0.8, color='black')
# plt.title('step vs pos&vel', fontsize=16)
# plt.xlabel('step', fontsize=14)
# plt.ylabel('', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout() # ラベルが重ならないように調整
# plt.show()

# GIFアニメーションの作成
imageio.mimsave('cartpole_pid.gif', frames, fps=30)

env.close()