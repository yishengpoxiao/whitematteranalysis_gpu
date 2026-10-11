# Atlas 距离融合与流水处理（2026-10-11）

相对上一轮 `b2ce685`，完成四项优化，并保留 float64 默认精度和 CPU 随机链。

## 改动

- 新增可选 `gpu/distance_kernel.py`。对于连续 float64 CUDA 张量和 1–32
  个采样点，融合正反向坐标距离、阈值、点归约与最小方向选择，只输出
  二维距离矩阵。支持 Mean/MeanSquared/Hausdorff；Mean 使用正确舍入的
  double sqrt，阈值常量显式为 float64，关闭 FMA，并匹配 PyTorch 的
  归约顺序。Hausdorff 保留 NaN 传播。
  float32、StrictSimilarity、更长采样或非连续输入保留原 PyTorch 实现；
  没有 Triton 时也保留原路径。现有 bilateral 逻辑不变。
- `_embed_to_rgb` 的两次逐纤维 colorsys 循环改为 NumPy 向量计算，保留
  分支优先级、逐步运算顺序、NaN 处理和输入修改约定。CPU 后端也受益。
- B-spline 每次优化持有一个 VTK 网格、float32 位移数组及系数滤波器；
  每次求值只更新数组和 VTK 修改时间。系数滤波仍在 CPU，保持原格式和
  舍入方式，不重新实现滤波算法。
- Atlas 用一个 CPU 工作线程执行原 scipy.vq 分类，与主线程的下一块 GPU
  计算重叠。最多保留一个待取分类任务，不累计 GPU 数据块；线程异常会
  传播给调用者。行和的 NumPy sqrt 仍在 CPU，以保持原有结果。
- 配准每块概率先存入 GPU 向量，全部完成后一次回传，减少逐块同步；
  floor、归一化、log 与熵求和仍在 CPU，保持优化轨迹。额外 GPU 常驻
  开销为每条 moving 纤维 8 字节。

## 验证

- 仓库单元测试 **20 项全部通过**。距离对拍覆盖 1/2/3/7/10/15/16/20/
  31/32/50 点，float64/float32，四种方法，阈值 0/0.1/2/100；另覆盖
  NaN、Inf、非连续输入和非默认 CUDA 设备。
- Atlas 四方法、bilateral、两种精度及分块尾部对拍通过；原有 OOM 缩块
  与终止检查通过。全部标签与分类距离均按原 CPU scipy.vq 计算。
- 颜色测试覆盖灰色、RGB 分量相等、六种颜色分支、NaN 和空输入，结果
  与旧 colorsys 实现逐位一致。
- 缓存 VTK 系数在 2/3/4/6/10 网格上连续更新正位移、负位移、零位移及
  再次非零位移，与每次重建的 VTK 系数逐位一致。坐标、非刚性与仿射
  优化轨迹及异常释放测试通过。
- 无 Triton：17 项通过、3 项按条件跳过；无 torch/Triton：7 项 CPU 测试
  通过、13 项 GPU 测试跳过。
- 现有 GPU 一致性和无 CUDA 的 CPU-fast 脚本均通过，源码编译通过。
- 真实 Slicer 5.2.2 六模块加载通过，其内 20 项测试通过（单卡环境的
  两项设备切换测试跳过），进程退出码 0。旧 widget 桩脚本引用已删除
  模块，未修复且未计为通过。
- 真实 **100000 条**输入，标签、颜色、double 嵌入和最终 VTP 文件与
  `b2ce685` **逐位/逐字节一致**。此结论针对已测输入，不保证任意数据
  与 CPU BLAS 的浮点舍入逐位相同。

## 单卡 RTX 4090 基准

脚本 `tests/bench_fused_atlas.py --full` 直接加载 `b2ce685` 源码对照。
距离、颜色、B-spline、概率归约和分类重叠取预热后 5 次中位数；完整 atlas
取预热后 3 次中位数。GPU 测量在计时前后同步，默认 float64。

| 项目 | b2ce685 | 本轮 | 比值 |
|---|---:|---:|---:|
| Mean 距离块 | 29.84 ms | 5.87 ms | 5.08× |
| MeanSquared 距离块 | 26.95 ms | 3.71 ms | 7.26× |
| Hausdorff 距离块 | 27.54 ms | 4.86 ms | 5.67× |
| 距离块峰值额外张量显存 | 1798–1837 MiB | 39–40 MiB | 降低约 97.8% |
| 100000 条颜色生成 | 452.4 ms | 29.7 ms | 15.2× |
| B-spline 变换，含系数准备和回传 | 1.958 ms | 1.558 ms | 1.26× |
| 16 块配准概率归约与熵 | 2.792 ms | 2.527 ms | 1.10× |
| 同用新距离 kernel，仅比较分类重叠 | 117.4 ms | 107.1 ms | 1.10× |
| 完整 100000 条 atlas 标记 | 4.056 s | 1.226 s | 3.31× |

距离块为 2048 queries × 2500 fixed × 15 点、阈值 2；B-spline 为
1000 条 × 10 点、6³ 网格；配准为 1000×1000 条、每块 64 条 moving。
分类重叠为 10000 条、ORG 官方 atlas。完整标记输入为
`/data/yijie/TractData/val_dsi_studio/250932.vtk`，atlas 为
`/data/software/ORG-Atlases/ORG-800FC-100HCP`。完整计时包含输入 DeepCopy、
距离、嵌入、CPU 分类、颜色及输出 polydata 构建，不含读盘、配准、离群
删除、分簇写盘；不能当作完整临床流水线耗时。

可选 Triton 首次使用有 JIT 开销；小任务及不同采样数/阈值需要考虑编译
成本，收益随数据规模和机器负载变化。CPU sqrt 回传同步仍存在，本轮以
分类重叠减少其对总耗时的影响，没有改用低精度计算。

复测：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 python tests/bench_fused_atlas.py --full
```

本次 Python：`/data/yijie/miniconda3/envs/wma/bin/python`。
日志：`/tmp/wma-next-{final-tests,final-bench,final-slicer,consistency,cpu,no-triton,no-torch}.log`。
本机 Slicer 通过源码链接使用新实现；已有自包含 `.sx` 安装包未重新打包。
