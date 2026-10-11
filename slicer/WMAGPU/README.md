# WMA GPU Slicer Extension

基于 `/data/yijie/wma-GPU/slicer` 的六模块界面，包含当前仓库的 GPU/CPU
后端、joblib 1.4.2 和 xlrd。无需安装独立 WMA；atlas 数据需自行选择。

## 安装与使用

1. 在 Slicer 5.2 的 **Extensions Manager → Install from file** 中选择
   `WMAGPU-Slicer-5.2-*.zip`，重启 Slicer。
2. 菜单 **Diffusion → WMA GPU** 中选择功能页。
3. 先用 Slicer 的 Add Data 加载 `.vtk/.vtp` 纤维，选择输入节点、atlas
   目录、输出目录及后端，再点击执行。

| 功能页 | 功能 |
|---|---|
| WMA Pipeline | 仿射配准 → atlas 标记 → 离群移除 → 原空间回变换 → 解剖束 |
| WMA Register to Atlas | 受试者向 atlas 仿射配准 |
| WMA Groupwise Registration | 群体仿射配准、输出 registration atlas |
| WMA Atlas Builder | 多受试者谱聚类构建 atlas |
| WMA Label From Atlas | 标记、可选离群移除及解剖束映射 |
| WMA Merge & MRML | 合并簇、输出 scene.mrml |

标记 atlas 目录应包含 `atlas.p`；官方 ORG atlas 或自行构建的 atlas 均可。
完整 Pipeline 还需要配准 atlas、簇参考文件及 `T_*.mrml` 解剖映射文件。
缺失可选映射时界面会提示。无 SlicerDMRI 时使用 Model 节点显示纤维；
安装 SlicerDMRI 后使用 FiberBundle 节点。

Auto 默认有可用 CUDA 时使用 GPU，否则使用 CPU accelerated。
GPU 需要在 **Slicer 自身 Python** 中安装与其 Python 版本兼容的 CUDA
PyTorch；Triton 是可选的进一步加速。安装包不包含 torch/CUDA，也不会
自动下载依赖。无 torch 即可使用 CPU accelerated；Original 为原 joblib
算法。显式选择 GPU 而 CUDA 不可用时会给出错误提示。
默认 float64；随机采样和分类保持 CPU 实现。

执行过程中会按阶段更新状态；长计算仍在 Slicer 主线程进行，重计算期间
界面响应可能变慢。当前 GUI 配准页提供仿射模式；后端的非刚性 API 可
脚本调用，未作为 GUI 选项。未修改原参考目录或用户已有 Slicer 设置。
如果 Additional module paths 中已有旧 WMA 六模块，安装新扩展前在
Slicer 设置中移除旧路径，以免模块重名。

## 开发与打包

源码入口是 `Modules/`，共享计算及 GUI 助手在 `_vendor/`，防止被当作独立
Slicer 模块扫描。纯 Python 打包不需要 Slicer SDK：

```bash
python slicer/WMAGPU/tools/package_extension.py \
  --slicer-version 5.2 --output /tmp/WMAGPU-Slicer-5.2.zip
```

脚本始终打包当前仓库的 `whitematteranalysis/`，不会拷贝 build/lib 或旧 `.sx`
中的后端，并拒绝覆盖已有包。安装包适配指定的 Slicer major.minor 目录；
本次实际验证 Slicer 5.2.2 / Linux，其他版本应重新指定版本打包和验证。

也提供标准 `CMakeLists.txt` Extension 工程，使用 Slicer 开发构建的
`Slicer_DIR` 配置后可执行 `cmake --build ... --target package`。
[官方 Extension 开发说明](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)。
该扩展供本地 Install from file 安装，尚未提交到公共 Extensions Index。

## 验证

使用 Slicer 5.2.2 / Linux 验证扩展管理器安装、六个模块加载、无 SlicerDMRI
时的节点选择、CPU/GPU 标记一致性、仿射配准、并行写盘及 MRML 导出。
合成数据还覆盖 atlas 构建→重载→标记的簇编号一致性、群体配准输出、
单文件 MRML 和显式 GPU 不可用时的错误处理。完整 ORG 解剖束 Pipeline
仍需提供实际 atlas 数据后验证；这些检查不代替真实病例验证。

可在已有 WMA Python 环境中运行有限规模的核心功能测试：

```bash
PYTHONPATH=. python -m unittest discover -s slicer/WMAGPU/Testing -p test_core.py -v
```

`Testing/install_extension.py` 通过 `WMA_EXTENSION_ARCHIVE`、
`WMA_EXTENSION_INSTALL_ROOT`、`WMA_EXTENSION_SETTINGS` 指定安装包、
独立安装目录及独立扩展设置文件。安装后启动新的 Slicer 进程，使用
`--disable-settings --additional-module-path <安装目录>/WMAGPU/lib/Slicer-5.2/qt-scripted-modules`
及 `--python-script Testing/verify_extension.py` 验证已安装代码；保留
`WMA_EXTENSION_INSTALL_ROOT` 环境变量。测试使用 `--no-main-window --ignore-slicerrc`，
无显示服务器时可通过 `xvfb-run -a` 启动。
