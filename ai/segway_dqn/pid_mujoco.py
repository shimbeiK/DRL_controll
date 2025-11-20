import mujoco
import mujoco.viewer
import numpy as np
import time

# --- 制御パラメータ (PID制御) ---
# ゲインはシミュレーション環境やモデルの質量によって
# 慎重にチューニング（調整）する必要があります。
KP = 200.0  # 比例ゲイン (P): 現在の傾きに反応
KI = 0.5    # 積分ゲイン (I): 過去の傾きの蓄積に反応
KD = 20.0   # 微分ゲイン (D): 傾く速度（ダンピング）に反応

# 目標角度 (0 = 垂直)
TARGET_ANGLE = 0.0

# --- PID制御用のグローバル変数 ---
integral_error = 0.0  # 積分誤差の蓄積
MAX_INTEGRAL = 30.0   # 積分値の最大値（アンチワインドアップ）

# --- モデルとデータ (グローバル) ---
model = mujoco.MjModel.from_xml_path('two_wheel_robot/scene.xml')
data = mujoco.MjData(model)

def get_pitch_angle(m, d):
    """
    シャーシの現在のピッチ角（傾き）をラジアン単位で取得します。
    """
    # シャーシの回転行列を取得
    chassis_id = m.body('chassis').id
    rot_matrix = d.xmat[chassis_id].reshape(3, 3)
    
    # 回転行列からオイラー角 (XYZ) を計算
    euler_angles = np.zeros(3)
    mujoco.mju_mat2Euler(euler_angles, rot_matrix, 'xyz')
    
    # このモデルでは X軸周りの回転 (euler_angles[0]) が
    # 前後の傾き（ピッチ）に対応します。
    pitch = euler_angles[0]
    return pitch

def get_pitch_velocity(m, d):
    """
    シャーシの現在のピッチ角速度を取得します。
    """
    # センサー 'frameangvel' (シャーシの角速度) のインデックス
    sensor_id = m.sensor('frameangvel').id
    start_addr = m.sensor_adr[sensor_id]
    
    # X軸周りの角速度を取得
    pitch_vel = d.sensordata[start_addr] # index 0 = X軸
    return pitch_vel

def controller(m, d):
    """
    PID制御コントローラ。
    """
    global integral_error # 蓄積した積分誤差を更新するため
    
    # シミュレーションのタイムステップ (dt) を取得
    dt = m.opt.timestep

    # 1. 現在の状態（センサー値）を取得
    current_angle = get_pitch_angle(m, d)
    current_velocity = get_pitch_velocity(m, d)

    # 2. 誤差（目標との差）を計算
    error_angle = TARGET_ANGLE - current_angle

    # --- P項 (Proportional) ---
    # 現在の誤差（傾き）に比例した力
    p_term = KP * error_angle
    
    # --- I項 (Integral) ---
    # 過去の誤差を蓄積
    integral_error += error_angle * dt
    # 積分値が大きくなりすぎるのを防ぐ (アンチワインドアップ)
    integral_error = np.clip(integral_error, -MAX_INTEGRAL, MAX_INTEGRAL)
    
    i_term = KI * integral_error
    
    # --- D項 (Derivative) ---
    # 傾く速度（角速度）に比例した力（ブレーキ・ダンピング）
    # 速度を抑える方向（エラーの微分とは逆）に働く
    d_term = KD * current_velocity
    
    # 3. 合計トルク（制御信号）を計算
    # P と I は傾きを戻す力、D は動きを抑える力 (マイナスで適用)
    control_signal = p_term + i_term - d_term

    # 4. 左右のモーターに同じトルクを適用
    d.ctrl[0] = control_signal  # left_motor
    d.ctrl[1] = control_signal  # right_motor

# --- メインの実行処理 ---
def main():
    global model, data, integral_error

    # XMLファイルを読み込む
    # xml_path = 'inverted_pendulum.xml'
    # try:
    #     model = mujoco.MjModel.from_xml_path(xml_path)
    # except Exception as e:
    #     print(f"Error loading XML: {e}")
    #     return

    data = mujoco.MjData(model)

    # MuJoCoビューアをパッシブモードで起動
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        start_time = time.time()
        
        while viewer.is_running():
            sim_time = data.time
            
            # リアルタイム実行のための待機
            while (time.time() - start_time) < sim_time:
                time.sleep(0.001)

            # 1. 制御器を呼び出して data.ctrl を設定
            try:
                controller(model, data)
            except Exception as e:
                print(f"Error in controller: {e}")
                break
            
            # 2. シミュレーションを1ステップ進める
            mujoco.mj_step(model, data)
            
            # 3. ビューアを同期（描画を更新）
            viewer.sync()
            
            # (オプション) もしロボットが倒れたらリセット
            current_angle = get_pitch_angle(model, data)
            if abs(current_angle) > 0.5: # 約30度以上傾いたら
                print("Fallen! Resetting simulation...")
                mujoco.mj_resetData(model, data)
                integral_error = 0.0 # 積分値もリセット
                start_time = time.time() # リアルタイム制御もリセット

if __name__ == "__main__":
    main()