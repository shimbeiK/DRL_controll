# DRL_controll（ai folder）
I will upload about Deep Reinforcement Learning with some robot controls!
詳細は[こちらのZenn](https://zenn.dev/shimbei/scraps/58fb288b134115)(下のZennページとは別)にまとめています．


# aiフォルダに含まれていないファイルについて
逆運動学を用いた４脚歩行ロボットの制御に関するプログラムをまとめている．
現在は運動学を用いた足先軌道の決定と，それに基づいた歩行をシミュレーション上で確認できている．
歩容についてはウォーク，クロール，ペース，トロット，バウンスを試した．

詳細は[こちらのZenn](https://zenn.dev/shimbei/scraps/426bdec27678f5)にまとめています．

### 各ファイルについて
・foot_orbit.py：足先軌道を２次元平面上で可視化するファイル．matplotlibを用いて可視化している．
・foot_pybullet.py：実際に４足動かして歩行の様子をシミュレーションしているファイル．物理エンジンはPybulletを用いている．
・parameters.json：４脚歩行ロボットの物理量をjson形式でまとめたファイル．このファイルとロボットモデルを変更すれば様々なロボットに対応できる．

### プログラムを実行するにあたって
詳細は省きますが１点注意があります．foot_pybullet.pyというファイルの中で外部ライブラリをインポートしていますので，各人でそちらもインポートする必要があります．
インストールについては[こちら](https://developers.agirobots.com/jp/foot-trajectory/)を参照して下さい．
