import gymnasium as gym
from gymnasium import spaces
import mujoco, json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class TwoWheeledInvertedPendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
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


    # def __init__(self, xml_file="two_wheel_robot/scene.xml", render_mode=None, max_step=5000):
    #     # super(TwoWheeledInvertedPendulumEnv(), self).__init__(gym.env)
    #     if not os.path.exists(xml_file):
    #         raise FileNotFoundError(f"XML file '{xml_file}' not found.")
    #     self.model = mujoco.MjModel.from_xml_path(xml_file)
    #     self.data = mujoco.MjData(self.model)

    def __init__(self, model, data, render_mode=None, max_step=parameters["max_step"]):
        # super(TwoWheeledInvertedPendulumEnv(), self).__init__(gym.env)
        self.model = model
        self.data = data
        self.frame_skip = 5
        self.max_step = max_step
        self.step_count = 0

        self.angle_threshold = np.pi/6  # radians
        self.pos_threshold = 0.8      # meters
        self.x_threshold = 0.8     # meters

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self.RENDING = 0
        self.MAX_TRQUE = 1  # 最大トルク
        self.LPF_ALPHA = 0.5  # ローパスフィルタの係数
        self.counter = 0
        self.filtered_torque = [0.0, 0.0]  # 各車輪の平滑化されたトルク値

        # 離散行動空間（例：11段階の速度）
        self.discrete_actions = np.linspace(-1.0, 1.0, self.parameters["action_size"])
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        # 観測空間：位置、速度、角度、角速度
        high = np.array([self.pos_threshold, np.finfo(np.float32).max,
                         self.angle_threshold, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _get_obs(self):
        mujoco.mj_step(self.model, self.data)
        x = self.data.xpos[1, 0]  # カート位置
        z = self.data.xpos[1, 2]  # カート位置
        x_dot = self.data.qvel[0]  # カート速度

        rotmat = self.data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        angle = rot.as_euler('xyz', degrees=False)[1]  # ロール角
        angle_dot = self.data.qvel[1]  # 回転の角速度

        return np.array([x, x_dot, angle, angle_dot, z], dtype=np.float32)

    def _reward(self, obs):
        x, x_dot, angle, angle_dot, z = obs
        reward = 2.0 - (2 * abs(angle) / self.angle_threshold) - 0.1 * (abs(x) / self.x_threshold)
        reward = float(np.clip(reward, -1.0, 1.0))
        return reward
    # def step(self, action):
    #     torque = self.MAX_TRQUE*self.discrete_actions[action-self.parameters["action_size"]//2]
    #     self.data.ctrl[0] = torque
    #     self.data.ctrl[1] = - torque
    #     self.counter += 1
    #     if(self.counter == 100):
    #         print("Applied torque:", torque)
    #         self.counter = 0

    #     for _ in range(self.frame_skip):
    #         mujoco.mj_step(self.model, self.data)
    #         # print(self.data.qpos)

    #     self.step_count += 1
    #     obs = self._get_obs()
    #     reward = self._reward(obs)

    #     done = bool(
    #         abs((self.data.xpos[0, 0]**2 + self.data.xpos[0, 1]**2)**(1/2)) > self.pos_threshold or
    #         abs(obs[2]) > self.angle_threshold or
    #         self.step_count >= self.max_step
    #     )
    #     if self.render_mode == "human" and self.RENDING == 1:
    #         pass

    #     return obs, reward, done, False, self.data, {}

    def step(self, action):
        # 🔹 目標トルク値
        target_torque = self.MAX_TRQUE * self.discrete_actions[action - self.parameters["action_size"] // 2]
        # print("Target torque:", target_torque)
        for i in range(self.frame_skip):
            # 🔹 LPF適用を各ステップで更新（より自然）
            self.filtered_torque[0] = (
                self.LPF_ALPHA * self.filtered_torque[0]
                + (1 - self.LPF_ALPHA) * target_torque
            )
            # if(i%3 == 0):
            #     print(self.filtered_torque[0])

            self.filtered_torque[1] = self.filtered_torque[0]

            # 🔹 平滑化後トルクを適用
            self.data.ctrl[0] = self.filtered_torque[0]
            self.data.ctrl[1] = self.filtered_torque[1]
            # self.data.ctrl[0] = target_torque
            # self.data.ctrl[1] = target_torque

            mujoco.mj_step(self.model, self.data)
                
        self.step_count += 1
        obs = self._get_obs()
        reward = self._reward(obs)

        done = bool(
            abs(np.linalg.norm(self.data.xpos[1, :2])) > self.pos_threshold
            or abs(obs[2]) > self.angle_threshold
            or self.step_count >= self.max_step
        )

        return obs, reward, done, False, self.data, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # self.data.qpos[0:2] = np.random.uniform(-0.1, 0.1, size=2)
        # self.data.qvel = np.random.uniform(-0.1, 0.1, size=self.model.nv)

        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self, render_mode):
        print("render called")
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            self.RENDING = 1

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.renderer = None

if __name__ == "__main__":
    print("実行するファイルが違うで。")