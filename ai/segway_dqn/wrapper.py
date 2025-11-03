'''
Gym環境に必要な要素
１．観測空間の正規化：   倒立振子の位置、速度、角度、角速度を観測

２．報酬調整：          現在の位置と速度から報酬を決定
                        ー 位置が中心から離れるほどペナルティ
                        ー 角度が大きいほどペナルティ

３．終了条件の調整 ：    一定回数が経つor位置・角度が一定値以上になる
                        ー 角度が±0.5ラジアン（約28.6度）以上
                        ー 台車の位置が±4.0メートル以上（前後）

４．離散化（for DQN）：  入力（台車の移動速度）を離散化する
                        ー 例：[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]の１1段階

５．観測のために行動：  入力を受けてシミュレーション環境を１ステップ進める
                        全てのステップを計算していると時間がかかる＋実空間では処理間隔が生じるためデータ量を間引く
                        ー １フレームスキップ

６．レンダリング設定

mainプログラムで呼び出すのはreset(), step(), close()のみ。それ以外はstep()に含まれる。
'''

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class TwoWheeledInvertedPendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # 初期設定
    def __init__(self, xml_file="two_wheel_robot/scene.xml", render_mode=None, max_episode_steps=500):
        # --- 1. MuJoCoモデルとデータの読み込み ---
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"XML file '{xml_file}' not found.")
        self.model = mujoco.MjModel.from_xml_path('two_wheel_robot/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # --- 2. 環境パラメータの設定 ---
        self.frame_skip = 2 # 1回のstepで進めるシミュレーションステップ数
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # 終了条件の閾値
        self.angle_threshold = 0.5  # ラジアン (約28.6度)
        self.x_threshold = 4.0      # メートル

        # --- 4. レンダリングの準備 ---
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

    # １．観測空間の正規化：   倒立振子の位置、速度、角度、角速度を観測
    def _get_obs(self):
        pos = self.data.xpos[1]  # ルートボディの位置
        rotmat = self.data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        euler = rot.as_euler('xyz', degrees=True)
        # 座標と角度をreturn
        return pos[1], euler[0]

# ２．報酬調整：          現在の位置と速度から報酬を決定
#                         ー 位置が中心から離れるほどペナルティ
#                         ー 角度が大きいほどペナルティ
    def _reward(self, obs):
        pass

    # ３．終了条件の調整 ：    一定回数が経つor位置・角度が一定値以上になる
    #                         ー 角度が±0.5ラジアン（約28.6度）以上
    #                         ー 台車の位置が±4.0メートル以上（前後）
    # ５．観測のために行動：  入力を受けてシミュレーション環境を１ステップ進める
    #                         全てのステップを計算していると時間がかかる＋実空間では処理間隔が生じるためデータ量を間引く
    #                         ー １フレームスキップ
    def step(self, action):
        pass

    """環境を初期状態(ランダム)にリセットする。"""
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Gymnasiumの乱数シード設定を継承
        # --- 1. シミュレーションデータのリセット ---
        mujoco.mj_resetData(self.model, self.data)        
        # --- 2. 初期状態にランダムな揺らぎを追加 ---
        pass

    """シミュレーションを可視化する。"""
    def render(self):
        if self.render_mode is None:
            return

        if self.viewer is None:
            # humanモードの場合、glfwベースのビューアを初期化
            from mujoco.viewer import launch_passive
            self.viewer = launch_passive(self.model, self.data)
        
        if self.render_mode == "human":
            self.viewer.sync()

        # elif self.render_mode == "rgb_array":
        #     # rgb_arrayモードの場合、ヘッドレスレンダラを使用
        #     if self.renderer is None:
        #         self.renderer = mujoco.Renderer(self.model)
        #     self.renderer.update_scene(self.data)
        #     return self.renderer.render()

    """環境を終了する。"""
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.renderer = None