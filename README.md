# Mask-Attention_by_PPO

ロボットアームを用いて把持対象物体を把持する．深層強化学習を用いて動作を決定し，Mask-Attentionを用いて視覚的説明を行う．

# ディレクトリ
```
Mask-Attention_by_PPO/
├── 3Dmodels    ：ARC2017 RGB-D Datasetの3Dモデルが格納されている．
├── docker      ：Dockerに関するファイルが格納されている．
├── repos       ：外部のライブラリを保存している．
├── target_item ：把持対象物体とする場合の外観画像が格納されている．
└── urdf        ：ARC2017 RGB-D DatasetのURDFファイルが格納されている．
```

# 実行方法
## Dockerの起動
Mask-Attention_by_PPO/で以下を実行し，docker imageを作成します．
```
bash docker/build.sh
```
作成したら以下を実行し，dockerに入ります．
```
bash docker/run.sh
```
このときrun.shの`--mount type=bind,source="/data1/results",target=/home/${USERNAME}/Mask-Attention_by_PPO/results`部分でマウントを行うので，`source="/data1/results"`を自分のフォルダに変更してください．