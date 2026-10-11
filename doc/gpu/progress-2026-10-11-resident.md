# 配准 GPU 数据复用（2026-10-11）

## 实现

`gpu.registration.AffineObjective` 为一次优化保存 float64 固定纤维与移动纤维
设备数据。`RegisterTractography.compute()` 在优化器开始前上传一次，通过
`objective_function` 复用，在 `finally` 中清除设备数据引用。每次新的
`compute()` 都重新创建，支持多尺度、不同受试者以及原位修改后的输入。
直接调用 `objective_function()` 时仍读取当前 NumPy 输入。

距离、概率、熵、VTK 矩阵构建、CPU 优化器和随机链均沿用上一版算法。
CPU 后端不会创建此设备对象。3000+1000 条纤维、每条 10 点时，常驻的输入
设备数据约 0.96 MB，优化完成或失败后不再由注册对象持有。

## 验证

- GPU 单元测试 4 项通过：原版 VTK 坐标/目标对拍，三个优化器完整求值序列与
  最终变换逐位一致，同一 NumPy 对象原位改动后的下一次优化正确刷新，异常
  退出释放引用，以及 CPU 优化路径对照原版逐位一致。
- CUDA 不可见时 CPU 测试通过，3 项 GPU 测试明确跳过。
- 现有 `test_gpu_consistency.py`、`test_cpu_fast.py` 全部通过。
- 编译检查通过；真实 Slicer 5.2.2 六模块均 SETUP OK，退出码 0。
- 前轮已发现的旧桩测试引用已删除模块问题未改动，也未计为通过。

本轮日志：`/tmp/wma-resident-{tests,no-cuda,consistency,cpu,slicer,real-bench}-2026-10-11.log`。
测试中优化器刻意使用较小求值预算；达到预算的警告不表示已经收敛。

## 性能

RTX 4090 单卡，float64，10 点/纤维。合成输入固定种子 42，预热后两种路径
交替测量各 30 次，取中位数，计时包含目标函数返回前的同步。

| fixed × moving | 上一版无状态 GPU | 常驻数据 GPU | 耗时降低 |
|---|---:|---:|---:|
| 100 × 100 | 0.997 ms | 0.883 ms | 11.4% |
| 3000 × 1000 | 9.907 ms | 9.480 ms | 4.3% |
| 10000 × 100 | 4.472 ms | 3.527 ms | 21.1% |

真实纤维 `/data/yijie/TractData/val_dsi_studio/250932.vtk`，固定种子 42 抽取
4000 条，3000 fixed + 1000 moving：单次求值中位数 9.872 → 9.414 ms（4.6%）；
Cobyla 30 次求值含设备初始化的一次计时 0.297 → 0.287 s（3.3%）。完整目标
序列和最终变换逐位一致。单次优化计时只作参考，不能推算整个多尺度管线收益。

可复测：

```bash
CUDA_VISIBLE_DEVICES=0 /data/yijie/miniconda3/envs/wma/bin/python \
  tests/bench_registration.py \
  --input /data/yijie/TractData/val_dsi_studio/250932.vtk
```

非刚性 B-spline 的变换应用仍在 CPU；本轮未重打包 Slicer `.sx`。
本机 Slicer 的源码符号链接会自动使用新实现。
