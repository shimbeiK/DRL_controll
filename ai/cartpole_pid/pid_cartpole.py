import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

# 環境の初期化
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()

# PIDゲインの設定
Kp_position = 2.0
Kd_position = 0.5
Kp_angle = 50.0
Kd_angle = 1.0

# 目標値の初期設定
target_position = 0.0     # カートの目標位置（動かせる）
target_angle = 0.0        # ポールの目標角度（常に直立）

# 記録用リスト
linear_vels = []
angular_vels = []
position_errors = []
angle_errors = []

# エピソードループ
done = False
time_step = 0
total_reward = 0

while not done:
    # 状態の取得
    pos, pos_dot, angle, angle_dot = state

    # 目標位置を時間で移動させる例（周期的に左右に振る）
    target_position = np.sin(time_step / 50.0) * 1.5  # -1.5〜+1.5の範囲で振動

    # 誤差の計算
    position_error = target_position - pos
    velocity_error = -pos_dot
    angle_error = target_angle - angle
    angular_velocity_error = -angle_dot

    # PID操作量の計算
    output = (
        Kp_position * position_error +
        Kd_position * velocity_error +
        Kp_angle * angle_error +
        Kd_angle * angular_velocity_error
    )

    # 行動の選択（操作量を離散化）
    action = 1 if output > 0 else 0

    # 行動の実行
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    time_step += 1

    # ログ記録
    state = next_state
    linear_vels.append(pos_dot)
    angular_vels.append(angle_dot)
    position_errors.append(position_error)
    angle_errors.append(angle_error)

    time.sleep(0.02)

# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(linear_vels, label='Cart Velocity', color='blue')
plt.plot(angular_vels, label='Pole Angular Velocity', color='red')
plt.plot(position_errors, label='Position Error', color='green')
plt.plot(angle_errors, label='Angle Error', color='orange')
plt.title("CartPole PID Control (State-wise Error)")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

env.close()
