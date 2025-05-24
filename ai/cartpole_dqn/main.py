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

現状の問題：actionが変更されない➡state = next_stateをするときに学習データと混ざっていた．

１エピソード内でもランダム方策を取り入れないと過学習（過度なバイアス）が生じるっぽい
ε減衰について
DQNってなん入力に対応できる？

パラメータの決定方法について聞きたいね．
あとはGitHubに公開するだけ
'''
import gymnasium as gym
import numpy as np
import time, json, torch, os
import matplotlib.pyplot as plt
from dqn_lib import N_Network, ReplayBuffer

# 環境設定
env=gym.make('CartPole-v1')
# env=gym.make('CartPole-v1', render_mode='human')
print(env.action_space.n)
total_reward = 0
(state,_)=env.reset()
time_step = 0
average = 20
reward_datas = []
# MODEL_SAVE_PATH = "dqn_Cpole_weights.pth"  # モデルの重みを保存するパス
MODEL_SAVE_PATH = "dqn_Cpole_weights_noisy.pth"  # モデルの重みを保存するパス

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
        change_epsilon = max(0.01, change_epsilon * 0.995)
        print(change_epsilon)

    #     # εの確率でランダムな行動を選択
        return env.action_space.sample()
    # 1-εの確率で最適方策に基づいて価値関数を更新
    else:
        state = torch.FloatTensor(state).unsqueeze(0) # PyTorchのTensorに変換＋バッチ形式に対応
        with torch.no_grad(): # 勾配計算を無効化：順伝搬してるだけで学習はしなくていいから
            # print(Qnet(state), Qnet(state).argmax())
            return Qnet(state).argmax().item()
        
def calc_TDerror(states, actions, next_states, rewards, dones):
    # QネットワークとターゲットネットワークからQ値を取得（TD誤差を求めるため）
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


# episodes回学習をする
for episode in range(parameters["episodes"]):
    # 環境のリセット
    state, _ = env.reset()
    time_step = 0 # 本質はtotal_rewardと同じだが分かり易くするために分けた
    episode_over = False
    noise = 0
    total_reward = 0

    while not episode_over:
        # 行動の実行
        action = select_action(state)  # 現在の状態から方策に従って行動を選択
        next_state, reward, terminated, truncated, info = env.step(action) # 選択された行動によって環境を更新
        episode_over = terminated or truncated

        # リプレイバッファに保存
        Replay_Buffer.add(state, action, reward, next_state, episode_over)
        
        # 外乱制御
        # if np.random.rand() <= noise:
        #     print("do")
        #     next_state[1] += np.random.normal(0, 500) # カート速度に外乱
        #     next_state[3] += np.random.normal(0, 500) # ポール角速度に外乱
        # if time_step != 0:
        #     noise = 0.01 * int(time_step % parameters["episodes"] * 0.01)
        #     # print(noise)
        #     print(int(time_step % parameters["episodes"] * 0.01))

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
        w = Qnet.state_dict()

    # 報酬データを保存
    reward_datas.append(total_reward)
    print(f"Episode: {episode}, Total Reward: {total_reward}, Time Step: {time_step}")

# try:
#     torch.save(Qnet.state_dict(), MODEL_SAVE_PATH)
#     print(f"モデルの重みを {MODEL_SAVE_PATH} に正常に保存しました。")
# except Exception as e:
#     print(f"モデルの重みの保存中にエラーが発生しました: {e}")


data_20 = []
plot_step = []
step = int(parameters["episodes"] / 20)
for i in range(step):
    data_temporaly = 0
    for j in range(average):
        data_temporaly += reward_datas[average*i + j]
    data_20.append(data_temporaly / average)
    plot_step.append(i*step)
        
# グラフの作成
plt.figure(figsize=(12, 6))

# 元の報酬データをプロット (任意)
plt.plot(reward_datas, label='every Total-Rewards', alpha=0.8, color='grey')
plt.plot(data_20, label='20times average', color='blue')
# グラフの装飾
plt.title('Total Reward vs 20times average', fontsize=16)
plt.xlabel('episode', fontsize=14)
plt.ylabel('Total Reward', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout() # ラベルが重ならないように調整

# グラフをファイルに保存
output_filename = "total_reward_lessy.png"
plt.savefig(output_filename)

print(total_reward)
env.close()

plt.show()