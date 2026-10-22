# 原精度权重分发说明

> **历史分发说明。** 以下描述早期 replacement 张量分发包，不是当前 GitHub
> 代码包的安装流程。当前代码包不含 `checkpoints/finetuned/`；请使用
> `python setup/download_weights.py` 下载已组装完整权重，详见
> [权重与依赖文档](docs/weights.md)。下述重建命令只有另行准备历史差量包后才能使用。

本压缩包保留 Gate3DGen 代码和完整项目页，但以精确 replacement 张量替代五个完整训练权重文件。
恢复模型的变化参数保持原 FP32，其余参数保持各自原始 FP32/BF16；GateB 保持原 BF16。
未转换 FP16、未量化，也不是浮点相减的 delta。
未变化的张量来自固定版本 FireRed 基座，并转换到各目标张量的原始 dtype。
重建时逐张量检查 shape、dtype、内容 SHA256；不承诺重写后的 .bin 文件序列化字节相同。

## 安装并恢复

先按原项目 README 安装 Python/模型依赖。在 Gate3DGen 目录中运行：

```bash
python setup/restore_original_weights.py --download-base
python setup/check.py --weights-only
```

FireRed 固定 revision 为 `3bc3f2a12722fd9883eb6357500de191d56baaf5`，完整下载且不做文件过滤。
已有相同基座时可以跳过下载：

```bash
python setup/restore_original_weights.py --firered /absolute/path/to/FireRed-Image-Edit-1.1
```

恢复目标为原项目默认路径：`checkpoints/router/checkpoint-6000/exported_model` 和
`checkpoints/purifier/checkpoint-15000`。保留原配置和 marker，已有完整权重会拒绝覆盖。
若中断，请保留现场并检查 `checkpoints/reconstructed_original`，不要把半成品当作恢复完成。
本脚本未提供覆盖或自动恢复中断的行为。

只验证逐张量重建而不写完整模型：

```bash
python setup/restore_original_weights.py --firered /path/to/base --verify-only
```

Restoration replacement 672 张量，28.55 GB；GateB replacement 67 张量，2.02 GB。
需另行准备基座下载、完整权重重建以及 3D 依赖的磁盘空间。没有携带 Conda 环境。
项目页里的图和 GLB 是既有展示产物，没有重新推理。

`checkpoints/finetuned/SOURCE_TARGET_VALIDATION.json` 记录本次对原 Gate3DGen
目标参数 shape/dtype/内容 hash 与差量 manifest 的全量核对及分片 SHA256 验证。
此前重建审计保存在同目录 `EXPORT_AUDIT.json`。完整大模型恢复文件的写入未在本次重新执行；
CPU 小型测试覆盖重建格式。没有重新运行 GPU 效果评测或全新环境安装。

CPU 小型测试复现：`PYTHONPATH=setup python -m unittest discover -s tests -p test_replacement_weights.py -v`。

第三方权重、素材和代码各自遵守原许可证，本打包不授予新的再分发权限。
