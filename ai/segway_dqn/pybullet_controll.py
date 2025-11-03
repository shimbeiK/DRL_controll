import pybullet as p
import math as m
import time, json
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from decimal import Decimal, getcontext

getcontext().prec = 10  # 計算精度を50桁に設定
# データを格納するリスト

pre_targetPosition = -1
reset_num = 1
model_name = "two_wheel_robot/model.urdf"

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF(model_name,startPos, startOrientation)
# ロボットの位置/姿勢を設定可能なデバッグ用のスライダーを定義
reset_button = p.addUserDebugParameter("reset_button", 1, 0, 0)
vel_0 = p.addUserDebugParameter("vel0", -10, 10, 3)
vel_1 = p.addUserDebugParameter("vel1", -10, 10, -3)
force_0 = p.addUserDebugParameter("force0", -10, 10, 3)
force_1 = p.addUserDebugParameter("force1", -10, 10, 3)
IDs = [vel_0, vel_1, force_0, force_1]
bef_IDs = [vel_0, vel_1, force_0, force_1]

while True:
    IDs = [
        p.readUserDebugParameter(vel_0),
        p.readUserDebugParameter(vel_1),
        p.readUserDebugParameter(force_0),
        p.readUserDebugParameter(force_1)
    ]
    if IDs != bef_IDs:        
        p.setJointMotorControl2(robotId, 0, p.VELOCITY_CONTROL, 
                                targetVelocity=IDs[0], force=IDs[2])
        p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, 
                                targetVelocity=IDs[1], force=IDs[3])
        bef_IDs = IDs
    p.stepSimulation()

    if(p.readUserDebugParameter(reset_button) == reset_num):
        p.removeBody(robotId)
        robotId = p.loadURDF(model_name, startPos, startOrientation)
        reset_num += 1

    time.sleep(1./240.)  # ここが重要！
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()
