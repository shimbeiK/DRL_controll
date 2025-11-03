import pybullet as p
import pybullet_data
import time

# PyBullet の初期化
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

# ロボットのロード
robot_id = p.loadURDF("two_wheel_robot/model.urdf", [0, 0, 0.1])

# 制御ループ
for _ in range(10000):
    # 左右のホイールにトルク制御（例：前進）
    p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=5, force=1)
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=1)
    p.stepSimulation()
    time.sleep(1./240.)
