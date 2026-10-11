# 配准逐点距离融合 kernel（2026-10-11）

## 实现

新增 `gpu/registration_kernel.py`，使用可选 Triton kernel 一次计算纤维正向与
反向的逐点平方距离，减少 dx/dy/dz 临时张量及独立算子启动。显式关闭 FMA，
保留 `(dx*dx + dy*dy) + dz*dz` 的逐步舍入；采样点求和、最小方向选择、exp、
固定纤维求和及熵仍使用原有 PyTorch 运算。默认仍是 float64。

实现依据 [Triton 官方 kernel 示例](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
的索引和掩码方式；本地 Triton NVIDIA 编译器的 `enable_fp_fusion=False`
同时控制 LLVM 和 ptxas 的浮点融合。实际舍入一致性由下列对拍验证。

使用一维 launch grid，不受二维 grid 的 moving 行数上限影响；启动时切换到
输入张量对应的 CUDA 设备，并恢复调用者当前设备。未安装 Triton 时自动使用
PyTorch 距离实现，不要求安装新的必需依赖，不改变 CPU 后端或随机链。

把每个 tile 的临时分配移入独立函数，让失败 tile 的张量引用随调用释放。
同时修复已有 OOM 重试问题：缩到一条 moving 纤维仍失败时抛出原异常，避免
无限重试。分块内存估算仍沿用此前保守值。

## 验证

- 单元测试 7 项全部通过（`CUDA_VISIBLE_DEVICES=0,1`）。
- 11 种采样点数：1/2/3/7/8/10/15/16/20/31/50；正反向逐点结果、概率和
  分块熵与 PyTorch 路径逐位一致，覆盖 tile 尾部和概率下溢。
- Cobyla/BFGS/Powell 求值序列和最终变换与无 Triton 路径逐位一致；输入刷新、
  正常/异常释放、非默认设备选择及 OOM 缩块和终止检查通过。
- 模拟 Triton 无法导入时 5 项通过、2 项按条件跳过；模拟 torch 与 Triton 均
  不可导入时 CPU 测试通过、6 项 GPU 测试按条件跳过。
- 现有 GPU 一致性与 CPU-fast 对照脚本全部通过；全部相关源码编译通过。
- 真实 Slicer 5.2.2 六模块加载通过；该进程内 Triton 可用，30 次 Cobyla
  新旧 kernel 的求值序列和最终变换逐位一致。

## 真实数据性能

基准提交 `406c585`，RTX 4090 单卡，float64，输入
`/data/yijie/TractData/val_dsi_studio/250932.vtk`。固定种子 42 抽取 4000 条，
3000 fixed × 1000 moving × 10 点。目标函数交替测量各 20 次，优化器各测
3 次，均报告中位数。函数返回前会回传 CPU 结果，计时包含同步。

| 操作 | 上一版 | 本轮 | 收益 |
|---|---:|---:|---:|
| 预热后完整 GPU 目标函数 | 9.414 ms | 2.627 ms | 3.58× |
| 峰值额外张量显存，含变换与距离 | 1167.60 MiB | 572.43 MiB | 降低 51.0% |
| 预热后 Cobyla 30 次求值，含数据初始化 | 0.2843 s | 0.0774 s | 3.67× |
| 预热后 Cobyla 默认 300 次预算，实际 34 次求值 | 0.3212 s | 0.0869 s | 3.70× |

本进程首次目标函数求值为 0.131 → 0.626 s，含运行时初始化和已有磁盘缓存
加载，不能视为无缓存编译耗时。首次使用有 JIT 开销，短的一次性任务可能
得不到上述收益。该性能是局部配准基准，不代表完整 ORG 管线同等加速。

复测（脚本直接读取基准提交的源码进行对比）：

```bash
CUDA_VISIBLE_DEVICES=0 /data/yijie/miniconda3/envs/wma/bin/python \
  tests/bench_registration_kernel.py \
  --input /data/yijie/TractData/val_dsi_studio/250932.vtk --maxfun 300
```

日志：`/tmp/wma-kernel-{tests,consistency,cpu,no-torch,no-triton,slicer,real-bench,real-300-bench}-2026-10-11.log`。
旧的 Slicer 桩测试仍引用已删除模块，本轮未修复且未计为通过；真实 Slicer
独立验证已通过。本轮未重打包 `.sx`；本机 Slicer 源码链接会自动使用新实现。
非刚性 B-spline 的变换应用仍在 CPU，此 kernel 也会加速其后续熵目标计算。
