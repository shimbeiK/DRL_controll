theta = {
    "theta1": 10,
    "theta2": 20,
    "theta3": 30
}

# ループでアクセス
for j in range(3):
    key = f"theta{1+j}"
    print(f"{key} = {theta[key]}")  # 値を取得
    theta[key] += 5  # 値を変更

# 結果確認
print(theta["theta1"])  # 15
print(theta["theta2"])  # 25
print(theta["theta3"])  # 35