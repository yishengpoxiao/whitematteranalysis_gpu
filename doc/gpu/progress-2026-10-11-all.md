# 四项剩余优化交付（2026-10-11）

基准为上一轮提交 `7a5e5b8`，原版算法参照 `/data/software/whitematteranalysis`。
默认 float64、随机采样、优化器及 atlas 格式保持现有约定。

## 实现

1. **B-spline 变换**：新增 `gpu/bspline.py`。每次优化缓存固定纤维、移动
   纤维的网格索引和三次权重；Triton 在 GPU 应用位移场，保留 VTK 的
   Zero-border、float32 系数和 float64 运算顺序，关闭 FMA。系数滤波仍
   使用 VTK CPU 实现，每次仅上传小系数网格。无 Triton 时使用 VTK
   `TransformPoints` C++ 批量接口，CPU 后端也受益；不再逐点调用 Python。
   参考 [VTK 9.1 官方实现](https://github.com/Kitware/VTK/blob/v9.1.0/Filters/Hybrid/vtkBSplineTransform.cxx)。
   修复原有非刚性 Cobyla 进度记录引用未导入 `resource` 的问题，并避免
   小于 10 次预算时进度取模除零。缓存随优化正常结束或异常退出释放。
2. **配准距离归约融合**：对于 1–32 个采样点，直接输出每对纤维的正反向
   平均平方距离，不生成 `(moving,fixed,points)` 中间张量。按本机
   PyTorch 2.4.1 `Reduce.cuh` 的 lane 累加及相邻归约顺序实现；除法按
   [PyTorch CUDA 标量除法实现](https://github.com/pytorch/pytorch/blob/v2.4.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
   使用 float64 倒数乘法。超过 32 点保留上一轮 kernel 和 PyTorch 归约；
   无 Triton 使用 PyTorch，后续方向选择、概率与熵不变。
3. **Atlas 流式嵌入**：每块相似度计算后立即规范化、嵌入、分类，CPU 仅
   保存最终结果；默认 2048 条输入纤维一块，可用现有 tile 设置覆盖。
   OOM 缩块到一条仍失败则抛出异常。atlas 常量只上传一次，仅计算最后
   `number_of_eigenvectors+1` 列特征向量，而非全部 2500 列。保留原有
   行和输出约定，并正确传递显式 `gpu_float32` 选项。
4. **聚类拆分和写盘**：拆分复用向量化 `filter.mask`，保留纤维顺序、
   原有属性转换及空簇。写盘保留 CPU-only fork 进程池，子进程直接继承
   只读 VTK 对象，仅通过任务队列传递索引和文件名，去掉 NumPy 序列化
   和 VTK 重建；同时保留 field data、拓扑与活动属性。子进程不访问 CUDA。
   输入对象在函数返回前不得修改。线程方案实测更慢，未采用。创建进程池
   时暂时禁用 stdin 并恢复，避免 Slicer PythonQt 的标准输入缺少 `close()`
   导致子进程启动失败。

## 验证

- `tests/test_gpu_registration.py`、`test_bspline.py`、`test_atlas_blocks.py`、
  `test_output_pipeline.py` 共 15 项通过，包括非默认 CUDA 设备、OOM 缩块
  与终止、优化器完整求值轨迹、异常释放、VTK 文件字节、属性及空簇。
- B-spline 网格分辨率 2/3/4/6/10，覆盖网格节点、ULP 邻点、边界外坐标，
  GPU、CPU 批量坐标与 VTK 逐点结果逐位一致；非刚性 Cobyla 100 次预算
  的求值序列与最终变换与无 Triton 路径逐位一致。
- 距离归约覆盖 1/2/3/7/8/10/15/16/20/31/32/50 点，正反向均值、概率
  和熵与 PyTorch 逐位一致；仿射 Cobyla/BFGS/Powell 轨迹对拍通过。
- Atlas 四种距离方法、bilateral、float32/float64、尾块、空结果对拍通过。
  真实 ORG 800 atlas 的 10000 条输入标签及最终 VTP 文件与上一版一致。
  嵌入 double 数值存在约 2e-14 的 GEMM 舍入差异；不宣称任意数据逐位相同。
- 现有 GPU 一致性和 CPU-fast 验证脚本全部通过；源码编译通过。
- 无 Triton：15 项中 13 项通过、2 项跳过。无 torch/Triton：CPU 检查通过，
  GPU 项按可用性跳过。
- 真实 Slicer 5.2.2 六模块加载及其内测试通过（单可见 GPU 时设备切换
  测试按条件跳过）。旧 widget 桩脚本引用已删除模块，未计为通过。

## 单卡 RTX 4090 性能

输入 `/data/yijie/TractData/val_dsi_studio/250932.vtk`，float64，种子 42。
B-spline：1000 条 × 10 点、6³ 网格；拆分/写盘：10000 条、22 簇；atlas：
10000 条、ORG-800FC-100HCP 的 2500 条 Nystrom 样本、10 维嵌入。
CPU 操作和 B-spline 取 3 次中位数，atlas 单次计时含转换、嵌入和分类。

| 项目 | 上一版 | 本轮 | 收益 |
|---|---:|---:|---:|
| B-spline 变换，含系数准备和 GPU 回传 | 39.15 ms | 1.81 ms | 21.6× |
| 无 Triton 的 VTK 批量变换 | 39.15 ms | 4.06 ms | 9.65× |
| GPU 配准目标，3000 fixed × 1000 moving × 10 点 | 2.672 ms | 2.373 ms | 1.13× |
| 配准峰值额外张量显存 | 572.43 MiB | 114.67 MiB | 降低 80.0% |
| Cobyla 30 次，含初始化 | 77.36 ms | 68.08 ms | 1.14× |
| Atlas 相似度至嵌入/分类 | 0.615 s | 0.396 s | 1.55× |
| Atlas 峰值额外张量显存 | 7124.59 MiB | 1879.58 MiB | 降低 73.6% |
| 聚类拆分，保留 point/cell data | 0.469 s | 0.086 s | 5.44× |
| 8 进程 VTP 写盘 | 0.430 s | 0.377 s | 1.14× |

完整原始 **100000 条**纤维仅执行 atlas 标记、生成内存中的输出 polydata：
**3.95 s**。不含配准、预处理、离群删除及 800 簇文件写盘，不能与旧文档的
83348 条完整流水线 21 s 指标直接比较。以上为局部基准，首次 JIT 有额外
开销；运行时负载、磁盘缓存和输入大小会影响收益。

复测：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 python tests/bench_remaining_pipeline.py --full
CUDA_VISIBLE_DEVICES=0 python tests/bench_registration_kernel.py --baseline 7a5e5b8
```

本次使用 `/data/yijie/miniconda3/envs/wma/bin/python`。日志在
`/tmp/wma-all-{final-tests,final-slicer-clean,consistency,cpu,no-triton,no-torch,pipeline-bench,reduction-bench}.log`。
本轮更新仓库源代码和测试；本机 Slicer 源码链接使用新实现，已有自包含
`.sx` 安装包未重新打包，分发安装包前需另行打包。
