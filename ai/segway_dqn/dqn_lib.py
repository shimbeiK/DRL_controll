import gymnasium as gym
import numpy as np
import time,random, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

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


# Q値を出力するためのニューラルネットワーク
class N_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(parameters["input_size"], 128) # 入力4次元 → 隠れ層128ユニットへの全結合層（例：観測が4次元のとき）
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, parameters["action_size"])

    # この関数を実行すれば深層学習が出来る
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class ReplayBuffer:
    # バッファーの定義，バッチサイズの代入
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size) # dequeはリストのようなもの

    #関数によっては__len__()という関数が内部で知らないうちに使われる)
    def __len__(self):
        return len(self.buffer)

    # フィードバック[カートの位置，カートの速度，ポールの角度，ポールの角速度]を取得しリプレイバッファに保存．
    # 溢れたら古いデータを削除．ここでの経験は[状態，行動，報酬，次の状態]の4つ
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done) # 経験をタプルに格納
        self.buffer.append(data)

    # バッファーからランダムにバッチサイズ分のデータを取得
    def sampling(self):
        datas = random.sample(self.buffer, self.batch_size)
        
        # データを分割してそれぞれの変数に格納
        # print(datas)
        state = torch.FloatTensor(np.stack([x[0] for x in datas]))
        action = torch.LongTensor(np.array([x[1] for x in datas]).astype(np.int64))
        reward = torch.FloatTensor(np.array([x[2] for x in datas]).astype(np.float32))
        next_state = torch.FloatTensor(np.stack([x[3] for x in datas]))
        done = torch.FloatTensor(np.array([x[4] for x in datas]).astype(np.int32))
        # print(state, action)
        return state, action, reward, next_state, done
    
# class DQN_Agent:
#     def __init__(self, buffer_size=10000, batch_size=128, gamma=0.99, epsilon=0.1):
#         self.input_size = parameters["input_size"]
#         self.action_size = parameters["action_size"]
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.batch_size = batch_size
#         self.Qnet = N_Network()

#         # Qネットワークの初期化
#         self.Tnet = N_Network()
#         self.Tnet.load_state_dict(self.Qnet.state_dict())
#         self.Tnet.eval()

#         # リプレイバッファの初期化
#         self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
#         # これをPyTorchのTensorに変換する
        
    
#     def define_policy(self):
#         pass

#     # ε-greedyポリシーに従って行動を選択
#     def select_action(self, state):
#         if np.random.rand() <= self.epsilon:
#             # εの確率でランダムな行動を選択
#             return self.env.action_space.sample()
        
#         # 1-εの確率で最適方策に基づいて価値関数を更新
#         else:
#             state = torch.FloatTensor(state).unsqueeze(0) # PyTorchのTensorに変換＋バッチ形式に対応
#             with torch.no_grad(): # 勾配計算を無効化
#                 return self.q_net(state).argmax().item()
            
    
#     def calc_loss(self, batch):
#         pass