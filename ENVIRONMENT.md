# 环境配置

建议使用 Linux 或 WSL2 和 Python 3.10。原始硬件、系统和包版本见 `environment.txt`。小样本演示支持 CPU，完整训练建议用 NVIDIA GPU。

在解压目录创建环境。GPU 用户按 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/#v290)选择与驱动匹配的安装命令，例如把 CPU 命令的索引地址换成 `https://download.pytorch.org/whl/cu130`。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -c "import cv2, torch, torchvision, ultralytics; from nuscenes.scripts import export_2d_annotations_as_json; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

`requirements.txt` 采用 nuScenes devkit 兼容的 NumPy 与 OpenCV 版本。原始服务器演示使用 `environment.txt` 中的实际环境。Ubuntu 若报告缺少 `libGL.so.1`，可安装系统包 `libgl1`。
