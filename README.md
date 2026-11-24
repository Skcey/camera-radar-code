# 视觉信息严重退化下前视相机与毫米波雷达融合的道路目标二维检测

## 提交包入口

- [环境配置](ENVIRONMENT.md)与 `requirements.txt` 提供安装步骤，`environment.txt` 记录原实验环境。
- [数据下载与预处理](DATA.md)提供 nuScenes mini、nuScenes 扩展集和 Astyx 的下载、目录及预处理命令。
- `runs/` 内含三个种子的相机与精修检查点和融合选分记录，`results/` 内含论文逐种子指标。
- 在 Linux 上可用 `sha256sum -c SHA256SUMS` 校验包内文件。

## 小样本 demo

直接打开 `demo_example/nuscenes-mini-real.jpg` 查看服务器运行结果。每行是一帧实际采集的 nuScenes mini 前视图像，配对该帧的前向毫米波雷达回波。图像在推理时施加论文的 σ=4 高斯模糊。从左到右为真值、纯相机、纯图像精修和本文的雷达补检。真值仅用于展示，样本按每个验证场景的首个含至少两类标注的帧选取，不根据预测结果选帧。

`demo_example/nuscenes-mini-real.json` 记录场景、样本编号、原始相机和雷达文件、投影点数量与各方法的输出框。预览直接打开即可。要重新推理，按 [nuScenes mini 预处理](DATA.md#nuscenes-mini)下载并准备数据，再在解压根目录运行：

```bash
python rescue_examples.py --device cpu --out demo_output/nuscenes-mini.jpg \
  --json-out demo_output/nuscenes-mini.json
```

## 使用随包权重重新评估

完成 mini 预处理后，重新计算三个种子的 σ=4 验证结果。结果写入 `runs/recheck/`。

```bash
python validate_expansion.py --prepared data/prepared --raw data/nuscenes \
  --out runs/recheck/nuscenes_mini.json --device cuda:0
```

在原实验服务器重新验证的结果见 `results/verification_mini.json`。真实数据 CPU demo 和此三种子验证命令均已在该服务器运行。

重新拟合精修头、训练相机、验证扩展集，以及 RRPN 和消融实验命令见下文。

论文实验报告人工高斯强模糊（σ=4 像素）的结果。`results/seed_42.json`、`seed_43.json` 和 `seed_44.json` 保存原始强模糊评估。相机骨干先在 nuScenes mini 的原始训练图像上微调。两个精修头共用相同的混合增强，包括原始、σ=2 和 σ=4 图像。

新版新增两个冻结验证集。nuScenes 从官方验证集取与 mini 完全不相交的 30 个场景、1,191 帧，评价汽车、行人和自行车。Astyx HiRes2019 使用 546 帧并仅评价汽车。新增数据只做推理，三个原训练种子的权重和分数系数直接复用。`results/expansion/` 包含逐种子指标、数据统计、划分核验和下载校验值。

论文表 VII 展示本文方法与 RRPN 候选规则在相同 nuScenes 验证帧和强模糊条件下的实测。复测沿用本项目的冻结相机与精修头结构，RRPN 的 Fast R-CNN 分类回归流程属于原论文的配置。公开分数及其评价协议见相关工作。`rrpn_tables.py` 从逐种子记录重建表 VII。

## 重建表图和 PDF

```bash
python strong_blur_report.py
python strong_blur_plot.py
python expansion_tables.py
python rrpn_tables.py
python ablation_tables.py
cd paper
tectonic paper.tex
```

也可用 XeLaTeX 编译两次。生成表格仅需 Python 标准库，绘图需要 NumPy 和 Matplotlib。图 3 已随源码提供，可用真实两帧 demo 重新生成。将 `paper/authors.example.tex` 复制为同目录的 `authors.tex` 后，按实际情况填写作者、学号、排序和贡献。

## 从服务器实验记录复核

原始实验为 `runs/rescue_42`、`runs/rescue_43` 和 `runs/rescue_44`。从包含本复现包代码的目录运行下面的命令，`--extract` 指向上述三个目录的父目录。

```bash
python strong_blur_report.py --extract /camera_radar_detection/runs
```

如需重新训练，先按 [nuScenes 下载说明](https://www.nuscenes.org/nuscenes#download) 获取 `v1.0-mini`。在有 NVIDIA GPU 的服务器装好 Python、PyTorch、Torchvision、nuScenes devkit、Ultralytics、OpenCV 和 pycocotools，然后在项目根目录运行。下例复用相机模型权重，精修训练协议与原实验一致。

```bash
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$PWD/.ultralytics"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$PYTHON_BIN" prepare.py --root data/nuscenes --out data/prepared
for seed in 42 43 44; do
  weights=runs/rgb/model.pt
  if [ "$seed" != 42 ]; then weights="runs/seed_${seed}/rgb/model.pt"; fi
  "$PYTHON_BIN" radar_rescue.py --weights "$weights" --seed "$seed" --out "runs/retrain/rescue_${seed}"
done
"$PYTHON_BIN" strong_blur_report.py --extract runs/retrain
"$PYTHON_BIN" rescue_examples.py --out paper/figures/severe_examples.jpg
"$PYTHON_BIN" radar_recall_probe.py --blur 4 --out results/proposal_coverage.json
```

相机检查点可用 `experiment.py rgb` 重建。下例为种子 42。其余两次使用种子 43 和 44 及独立输出目录。若要使用新训练的相机检查点，把上面的精修命令中 `--weights` 指向相应新文件。

```bash
mkdir -p weights
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o weights/yolo11n.pt
python experiment.py rgb --weights weights/yolo11n.pt --seed 42 \
  --epochs 20 --batch 16 --workers 4 --amp --out runs/retrain/seed_42/rgb
```

## 新增数据的制备与冻结验证

- [nuScenes 官方下载入口](https://www.nuscenes.org/nuscenes#download) 提供训练验证元数据及关键帧。本实验使用 `v1.0-trainval_meta.tgz` 和 `v1.0-trainval{01,02}_keyframes.tgz`。解压元数据后，从两个关键帧包只解压 `samples/CAM_FRONT/*` 与 `samples/RADAR_FRONT/*`。
- [Astyx HiRes2019 数据归档](https://github.com/under-the-radar/radar_dataset_astyx) 提供配对雷达、相机、标定及三维框。只用 `dataset_astyx_hires2019/`。由于传感器字段定义不同，跨数据集推理将 RCS 和自车补偿速度通道置零，使用实际投影和深度。三类 mAP 和仅汽车 AP 分开报告。

沿用上一节的服务器环境变量，并在包含 `runs/rgb`、`runs/seed_43/rgb`、`runs/seed_44/rgb` 和 `runs/rescue_*` 原始权重的项目根目录运行。

```bash
mkdir -p runs/expansion
"$PYTHON_BIN" prepare.py --version v1.0-trainval \
  --root data/expansion/nuscenes --out data/expansion/nuscenes_val_prepared \
  --splits val --allow-subset --exclude-mini
"$PYTHON_BIN" audit_expansion.py --raw data/expansion/nuscenes \
  --prepared data/expansion/nuscenes_val_prepared \
  --out runs/expansion/nuscenes_split_audit.json
"$PYTHON_BIN" validate_expansion.py --raw data/expansion/nuscenes \
  --prepared data/expansion/nuscenes_val_prepared \
  --out runs/expansion/nuscenes_unseen.json
"$PYTHON_BIN" prepare_astyx.py --raw /path/to/dataset_astyx_hires2019 \
  --out data/expansion/astyx_prepared
"$PYTHON_BIN" validate_expansion.py --prepared data/expansion/astyx_prepared \
  --out runs/expansion/astyx.json
"$PYTHON_BIN" expansion_tables.py --results runs/expansion
```

## RRPN 候选规则复测

锚框参数和顺序取自 [RRPN 作者代码](https://github.com/mrnabati/RRPN/tree/c9f4c291e53e64383158542ba79bf99d1cecebdc)，修订号为 `c9f4c291e53e64383158542ba79bf99d1cecebdc`，许可证在 `RRPN-LICENSE.txt`。使用同一批前向雷达回波及对应的相机深度，先在 nuScenes 原图坐标生成候选，再裁剪并映射到 640×640 坐标。精修头重新拟合，使用相同的训练轮数和选分方法。两套 nuScenes 验证集均仅评价 σ=4。复现包的 `results/rrpn/` 保存参考结果。公开报告分别来自 [Nobis 等表 II](https://arxiv.org/pdf/2005.07431)、[Nabati 和 Qi 表 I](https://arxiv.org/pdf/2009.08428) 和 [Stäcker 等表 1](https://www.dfki.de/fileadmin/user_upload/import/11895_Fusion_Point_Pruning_for_Optimized_2D_Object_Detection_with_Radar_Camera_Fusion.pdf)。

沿用前面的环境和数据目录，在服务器项目根目录执行。

```bash
mkdir -p results/rrpn
for seed in 42 43 44; do
  weights=runs/rgb/model.pt
  if [ "$seed" != 42 ]; then weights="runs/seed_${seed}/rgb/model.pt"; fi
  "$PYTHON_BIN" rrpn_reference.py --seed "$seed" --weights "$weights" \
    --external-data data/expansion/nuscenes_val_prepared \
    --external-raw data/expansion/nuscenes --out "runs/rrpn_reference/$seed"
  cp "runs/rrpn_reference/$seed/metrics.json" "results/rrpn/seed_${seed}.json"
done
"$PYTHON_BIN" rrpn_tables.py
```

## 模块消融实验

消融使用原来的相机及完整、纯图像精修头作为参照。各消融变体使用相同训练划分重拟合，按训练集内的两个开发场景选择分数系数。移除质量加权时，由于直接使用分类概率，额外在开发集搜索更小的系数。只在 σ=4 条件下评价，未见场景和 Astyx 的标注仅用于计算指标。

沿用前述环境变量和三个数据目录，在项目根目录运行。

```bash
mkdir -p results/ablation
for seed in 42 43 44; do
  weights=runs/rgb/model.pt
  if [ "$seed" != 42 ]; then weights="runs/seed_${seed}/rgb/model.pt"; fi
  "$PYTHON_BIN" module_ablation.py --seed "$seed" --weights "$weights" \
    --rescue "runs/rescue_$seed" --out "runs/module_ablation/$seed"
  cp "runs/module_ablation/$seed/metrics.json" "results/ablation/seed_$seed.json"
done
"$PYTHON_BIN" ablation_tables.py
```
