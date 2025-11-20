'''
・Envの調整
・DQNを導入
・行動と状態をDQNと合併

１．action:角速度を与えることで変化を求める（最初は位置変化を与えるだけ）
２．actionによって得られたモデルを描画し、位置と速度を得る
３．フィードバックする（正しい報酬が出力されているか確認する）
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
left_idx = 0
right_idx = 1

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
render_mode = "human"
env.render(render_mode=render_mode)
# env=gym.make('CartPole-v1', render_mode='human')
total_reward = 0
# print(env.reset())
(state,_)=env.reset()
time_step = 0
average = 20
reward_datas = []
MODEL_SAVE_PATH = "mujoco_Cpole_weights.pth"  # モデルの重みを保存するパス
# MODEL_SAVE_PATH = "dqn_Cpole_weights_noisy.pth"  # モデルの重みを保存するパス
Qnet = N_Network()
# Qnet.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu")) # 学習済みモデルの重みを読み込む
Tnet = N_Network()
Tnet.load_state_dict(Qnet.state_dict())
Tnet.eval()
Replay_Buffer = ReplayBuffer(parameters["buffer_size"], parameters["batch_size"])
optimizer = torch.optim.Adam(Qnet.parameters(), lr=parameters["lr"])

change_epsilon = parameters["epsilon"] # ε-greedy法の初期値  
# ε-greedyポリシーに従って行動を選択
 

def select_action(state):
    global change_epsilon
    # if np.random.rand() <= parameters["epsilon"]:
    if np.random.rand() <= change_epsilon:
        change_epsilon = max(0.01, change_epsilon - 1e-4)
        # print(change_epsilon)

    #     # εの確率でランダムな行動を選択
        return np.random.randint(0, parameters["action_size"])
    # 1-εの確率で最適方策に基づいて価値関数を更新
    else:
        state = torch.FloatTensor(state).unsqueeze(0) # PyTorchのTensorに変換＋バッチ形式に対応
        with torch.no_grad(): # 勾配計算を無効化：順伝搬してるだけで学習はしなくていいから
            # print(Qnet(state), Qnet(state).argmax())
            return Qnet(state).argmax().item()
        
def calc_TDerror(states, actions, next_states, rewards, dones):
    # QネットワークとターゲットネットワークからQ値を取得（TD誤差を求めるため）
    # print(actions)
    actions = torch.LongTensor(actions).unsqueeze(1)  # [batch_size] → [batch_size, 1]
    q_values = Qnet(states) # → Q(s_t, ·) 全アクションのQ値を出力（形状：[batch_size, action_dim]）
    # print(q_values, action)
    q_value = q_values.gather(1, actions).squeeze() # actionに対応するQ値を出力（形状：[batch_size]

    with torch.no_grad():
        next_q_max = Tnet(next_states).max(1)[0]
        q_value_target = rewards + (1 - dones) * parameters["gamma"] * next_q_max # Q値のターゲット値をベルマン方程式 に従って計算

    # Q値から価値関数とTD誤差を計算．TD誤差を最小化するようにQネットワークのパラメータを更新する
    loss = torch.nn.MSELoss()(q_value, q_value_target)

    # 深層学習のルールに従ってQネットワークのパラメータを更新
    optimizer.zero_grad() # 勾配を初期化
    loss.backward() # 誤差逆伝播法で勾配を計算
    optimizer.step() # Qネットワークのパラメータ(重み)を更新

def disturbance_control(noise, next_state, time_step):
    # 外乱制御の実装（必要に応じて）
    if np.random.rand() <= noise:
        print("do")
        next_state[1] += np.random.normal(0, 500) # カート速度に外乱
        next_state[3] += np.random.normal(0, 500) # ポール角速度に外乱
    if time_step != 0:
        noise = 0.01 * int(time_step % parameters["episodes"] * 0.01)
        # print(noise)
        print(int(time_step % parameters["episodes"] * 0.01))

# episodes回学習をする
print("ok")
for episode in range(parameters["episodes"]):
    # 環境のリセット
    state, _ = env.reset()
    time_step = 0 # 本質はtotal_rewardと同じだが分かり易くするために分けた
    episode_over = False
    noise = 0
    total_reward = 0
    if(render_mode == "cpu"):
        while not episode_over:
            # 行動の実行
            action = select_action(state)  # 現在の状態から方策に従って行動を選択
            next_state, reward, terminated, truncated, data , info = env.step(action) # 選択された行動によって環境を更新
            episode_over = terminated or truncated

            # リプレイバッファに保存
            Replay_Buffer.add(state, action, reward, next_state, episode_over)
            
            # 外乱制御
            # disturbance_control(noise, next_state, time_step)

            state = next_state
            if len(Replay_Buffer) >= parameters["batch_size"]:
                # バッファに十分データが溜まっていれば，バッファからバッチサイズ分をサンプリング
                states, actions, rewards, next_states, dones = Replay_Buffer.sampling()
                # Qネットワークの学習
                calc_TDerror(states, actions, next_states, rewards, dones)
                
            # 一定のtime_stepごとターゲットネットワークの重みを更新
            if time_step % parameters["td_interval"] == 0:
                Tnet.load_state_dict(Qnet.state_dict())
            
            # print(next_state, reward, terminated, truncated, info)
            total_reward += reward
            time_step += 1
            # w = Qnet.state_dict()

        # 報酬データを保存
        reward_datas.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Time Step: {time_step}")

    if(render_mode == "human"):
        if(episode % 30 == 0):
            with mujoco.viewer.launch_passive(model, data) as viewer:
                step_start = time.time()
                print("2秒後に開始します")
                time.sleep(2)

                while not episode_over:
                    # 行動の実行
                    action = select_action(state)  # 現在の状態から方策に従って行動を選択
                    next_state, reward, terminated, truncated, data , info = env.step(action) # 選択された行動によって環境を更新
                    episode_over = terminated or truncated

                    # リプレイバッファに保存
                    Replay_Buffer.add(state, action, reward, next_state, episode_over)
                    
                    state = next_state
                    if len(Replay_Buffer) >= parameters["batch_size"]:
                        # バッファに十分データが溜まっていれば，バッファからバッチサイズ分をサンプリング
                        states, actions, rewards, next_states, dones = Replay_Buffer.sampling()
                        # Qネットワークの学習
                        calc_TDerror(states, actions, next_states, rewards, dones)
                        
                    # 一定のtime_stepごとターゲットネットワークの重みを更新
                    if time_step % parameters["td_interval"] == 0:
                        Tnet.load_state_dict(Qnet.state_dict())
                    
                    # print(next_state, reward, terminated, truncated, info)
                    total_reward += reward
                    time_step += 1
                    # w = Qnet.state_dict()

                    viewer.sync()
                    rotmat = data.xmat[1].reshape(3, 3)
                    rot = R.from_matrix(rotmat)
                    euler = rot.as_euler('xyz', degrees=True)

                    # print("pos-data:",data.xpos[1], data.qpos[1])
                    # print("angle-data:",euler)
                    # print("angle-vel:",data.qvel[1])
                    # print("Total Reward:", total_reward)
                    # print("Time Step:", time_step)

                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    # print(model.opt.timestep, time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            while not episode_over:
                # 行動の実行
                action = select_action(state)  # 現在の状態から方策に従って行動を選択
                next_state, reward, terminated, truncated, data , info = env.step(action) # 選択された行動によって環境を更新
                episode_over = terminated or truncated

                # リプレイバッファに保存
                Replay_Buffer.add(state, action, reward, next_state, episode_over)
                
                # 外乱制御
                # disturbance_control(noise, next_state, time_step)

                state = next_state
                if len(Replay_Buffer) >= parameters["batch_size"]:
                    # バッファに十分データが溜まっていれば，バッファからバッチサイズ分をサンプリング
                    states, actions, rewards, next_states, dones = Replay_Buffer.sampling()
                    # Qネットワークの学習
                    calc_TDerror(states, actions, next_states, rewards, dones)
                    
                # 一定のtime_stepごとターゲットネットワークの重みを更新
                if time_step % parameters["td_interval"] == 0:
                    Tnet.load_state_dict(Qnet.state_dict())
                
                # print(next_state, reward, terminated, truncated, info)
                total_reward += reward
                time_step += 1


        # 報酬データを保存
        reward_datas.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Time Step: {time_step}")


try:
    torch.save(Qnet.state_dict(), MODEL_SAVE_PATH)
    print(f"モデルの重みを {MODEL_SAVE_PATH} に正常に保存しました。")
except Exception as e:
    print(f"モデルの重みの保存中にエラーが発生しました: {e}")