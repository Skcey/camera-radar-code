# 数据下载与预处理

按发布方的下载流程和许可获取真实数据。提交包提供训练权重、指标及两帧真实数据的推理预览。重新运行 demo 时需要下载 nuScenes mini 原始数据。

## nuScenes mini

在 [nuScenes 下载页](https://www.nuscenes.org/nuscenes#download) 注册并登录，下载 v1.0-mini 数据和元数据。将以下示例中的文件路径换成实际下载路径。

```bash
mkdir -p data/nuscenes
tar -xzf /path/to/v1.0-mini.tgz -C data/nuscenes
python prepare.py --root data/nuscenes --out data/prepared
```

解压后 `data/nuscenes/` 应直接包含 `v1.0-mini/`、`samples/` 和 `maps/`。预处理使用 `samples/CAM_FRONT/` 与 `samples/RADAR_FRONT/`。输出 `data/prepared/summary.json` 应记录 323 个训练帧、81 个验证帧。

## nuScenes 未见场景

在同一官方页面取得 `v1.0-trainval_meta.tgz`、`v1.0-trainval01_keyframes.tgz` 和 `v1.0-trainval02_keyframes.tgz`。以下命令假定压缩包放在仓库根目录。

```bash
mkdir -p data/expansion/nuscenes
tar -xzf v1.0-trainval_meta.tgz -C data/expansion/nuscenes
for archive in v1.0-trainval01_keyframes.tgz v1.0-trainval02_keyframes.tgz; do
  tar -xzf "$archive" -C data/expansion/nuscenes --wildcards 'samples/CAM_FRONT/*' 'samples/RADAR_FRONT/*'
done
python prepare.py --version v1.0-trainval --root data/expansion/nuscenes \
  --out data/expansion/nuscenes_val_prepared --splits val --allow-subset --exclude-mini
```

目标为 30 个未与 mini 重合的场景、1,191 帧。对于发布方提供并支持 HTTP Range 的授权链接，可用 `fetch_keyframes.py` 续传。

若元数据包解压后缺少 `maps/`，从官方地图包补齐原始栅格地图，再运行预处理。

## Astyx HiRes2019

下载或克隆 [Astyx 数据归档](https://github.com/under-the-radar/radar_dataset_astyx)。只用 `dataset_astyx_hires2019/`，共 546 帧，仅评价汽车。

```bash
git clone https://github.com/under-the-radar/radar_dataset_astyx.git data/astyx-source
python prepare_astyx.py --raw data/astyx-source/dataset_astyx_hires2019 \
  --out data/expansion/astyx_prepared
```
