# SWE：位移法浅水动力学引擎

Python 接口，NumPy CPU 基准后端和 NVIDIA Warp CUDA 后端。移植自相邻目录
`../swe3d_minimal` 的已验证 MATLAB 核心，采用水平拉格朗日三角形网格、三点
Hammer 积分、集中质量和显式 Verlet 推进。水底高程向上为正，使用 SI 单位。

这是**非色散、深度平均浅水模型**；三维显示由水平位置、水深和地形重建，
没有独立求解竖向速度。版本控制在本目录的 `.git` 中，原 MATLAB 和论文未修改。

## 作为包导入到自己的程序

安装包名为 `dut-swe-engine`，代码中使用 `import swe`。
从 [examples/basic.py](examples/basic.py) 开始：该文件自己构建网格、地形和
初始水面，并显式调用 `swe.DisplacementSolver.step()`，展示应用如何使用引擎。

三个独立应用示例分别为 [平滑波](examples/standing_wave.py)、
[水盆晃荡](examples/sloshing.py) 和 [绕岛](examples/island.py)。它们使用
`swe.ModelBuilder` 建模，通过公开的 `swe.run()` 运行，可选择 CPU、CUDA
和可视化，不依赖内置算例函数或 CLI。示例可复制到安装了引擎的其他项目。

```sh
.venv/bin/python examples/basic.py
.venv/bin/python examples/sloshing.py --device cpu
.venv/bin/python examples/sloshing.py --device cpu --viewer viser --speed 0.5 --loop --vertical-scale 5
```

安装、改参数和应用职责说明见 [examples/README.md](examples/README.md)。

## 本机直接运行

项目已建立 `.venv`。在本目录运行：

```sh
# 原绕岛算例，CPU，无界面
.venv/bin/swe --case island --device cpu

# 简单桌面窗口；计算结束后保留窗口
.venv/bin/swe --case island --viewer pyvista --hold

# 平滑驻波、抛物面水盆晃荡
.venv/bin/swe --case wave --viewer pyvista --hold
.venv/bin/swe --case sloshing --viewer pyvista --hold

# 浏览器查看；在本地打开 http://127.0.0.1:8080
.venv/bin/swe --case island --viewer viser --paused --hold

# CPU 晃荡动画：半速循环，竖向显示放大 5 倍
.venv/bin/swe --case sloshing --device cpu --viewer viser --speed 0.5 --loop --vertical-scale 5
```

桌面快捷键：空格暂停/继续，`n` 单步，`r` 重置，`1/2/3` 切换水面/水深/速度，
`w` 网格边，`t` 俯视，`i` 三维视角，`q` 退出。浏览器使用侧边控件。
浏览器中的标量采用**节点颜色**叠加简单水面，桌面窗口提供连续的顶点颜色插值。
底形显示使用独立的稀疏采样网格，展示干岛；该显示网格不参与数值求解。

交互查看器默认按真实时间播放；`--speed 0.5` 是半速，`--speed 2` 是两倍速。
算力不足时播放会变慢，物理步长不会因此增大。`--loop` 在每段结束后保留终帧
0.5 秒，再从初态重新计算；时间归零表示新一轮播放，并非物理状态连续衔接。
`--vertical-scale 5` 仅将水面和水底的显示高度放大 5 倍，色标与诊断仍为真实值。
`--fps 25` 只限制显示更新频率；`--dt` 是物理求解步长。无界面运行不限制速度。
`--field speed --clim 0 0.5` 设置固定色标，
`--output outputs/state.npz` 保存最后状态。默认不保存每一步的全场历史。

## 安装与平台

推荐 Python 3.12 或 3.13。CPU 核心只依赖 NumPy，图形与 CUDA 都是可选依赖。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .                 # CPU
.venv/bin/python -m pip install -e '.[cuda]'          # NVIDIA CUDA + Warp CPU
.venv/bin/python -m pip install -e '.[desktop,web]'   # 可选查看器
.venv/bin/python -m pip install -e '.[dev]'           # 回归测试
```

项目同时提供 `uv.lock`；使用 uv 时可运行 `uv sync --extra dev --extra desktop --extra web`。
Warp 固定为已测试的 `1.17.0`。Apple Silicon Mac 使用 CPU 求解，不支持 CUDA；
图形显示仍可独立使用机器的图形设备。平台要求见
[Warp 安装文档](https://nvidia.github.io/warp/stable/user_guide/installation.html)。

- `device="auto"`：初始化时有 CUDA 则使用 CUDA，否则使用 NumPy CPU，并可读取 `model.backend_reason`。
- `device="cpu"`：强制 NumPy，无需安装 Warp。
- `device="cuda"` / `"cuda:0"`：强制 GPU，不可用时明确报错。
- `device="warp:cpu"`：在 CPU 上执行相同 Warp 核函数，主要用于验证。

数值错误或运行中的 GPU 故障不会被自动解释成“CUDA 不可用”并悄悄回退。
第一版只接受 `float64`，CUDA 的并行归约采用误差容限验证，不保证逐位一致。

## Python 接口

```python
import numpy as np
import swe

mesh = swe.Mesh.rectangle(nx=80, ny=16, size=(10.0, 2.0), walls=True)
builder = swe.ModelBuilder(gravity=9.81, density=1000.0, manning=0.0)
builder.add_water(
    mesh=mesh,
    bathymetry=swe.FlatBottom(z=0.0),
    surface_height=lambda xy: 1.0 + 0.001*np.cos(np.pi*xy[:, 0]/10.0),
    velocity=(0.0, 0.0),
)
model = builder.finalize(device="auto")
solver = swe.DisplacementSolver(model)
state, next_state = model.state(), model.state()

for _ in range(1000):
    solver.step(state, next_state, dt=0.001)
    state, next_state = next_state, state

snapshot = state.to_numpy()   # 显式复制到 CPU；不要在每个 GPU 步都调用
diagnostics = model.diagnostics(state)
state.save("outputs/checkpoint.npz")
restored = model.load_state("outputs/checkpoint.npz")
# 在另一台机器或另一后端，不必重跑初值函数：
restored_model, restored = swe.load_checkpoint("outputs/checkpoint.npz", device="cpu")
```

`Model` 定义几何、参数和初始场，建立后不可修改；修改场景应重新构建模型。
`State` 只保留当前位移、速度、时间和步数。求解器的临时工作区属于模型的后端，
同一个模型不能同时在多个线程执行求解；并行场景应各自创建模型。

`snapshot` 包含 `points`（当前三维水面节点）、`displacement`、`velocity`、
`surface_height`、`depth`、`bed`、`element_jacobian`、`time`、`step_index`。
节点水深/水面是相邻单元重建结果的平均，不能替代单元积分点诊断。
`model.diagnostics(state)` 返回能量、体积、积分点最小水深、最小/最大 Jacobian 和最大速度。
体积一致性内建于材料描述，是实现检查，不是独立精度证明。

检查点包含模型指纹、地形、预计算几何和初始场采样数据，可用 `load_checkpoint`
直接在 CPU/CUDA 间恢复，不需要序列化 Python 函数。`model.load_state` 用于
模型指纹完全一致的已有模型；跨平台重新计算 sin/exp 可能改变末位，跨平台请使用
保存了预计算数组的 `load_checkpoint`。`state.reset()` 恢复初态。

## 地形、初值与网格约定

水底是高程 `b(x,y)`，初始水面是高程 `eta0(x,y)`，初始水深 `h0=eta0-b`。
内置 `FlatBottom`、`GaussianBottom`、`ParabolicBottom`、`GridBottom`。
任意 Python 地形函数可显式采样成支持 CPU/CUDA 的规则栅格：

```python
bottom = swe.GridBottom.from_function(
    lambda xy: 0.1*np.sin(xy[:, 0])*np.cos(xy[:, 1]),
    bounds=(-2.0, 12.0, -2.0, 4.0),
    shape=(257, 129),
)
```

地形栅格采用双线性插值及该插值的解析梯度；采样分辨率引入单独的地形误差。
地形范围必须覆盖**移动后**的节点和积分点，越界会拒绝时间步，不外推。
`surface_height` 接受标量、节点数组或 `function(xy)->(N,)`；节点数组在单元内
线性插值，而函数会在积分点直接计算。`velocity` 接受常量二元组、`(N,2)` 数组
或同形状函数。所有初始场只在建模时求值，随后在设备上推进。

自定义网格：`Mesh(points, triangles, fixed)`，水平 `points` 为 `(N,2)`，
`triangles` 为零起始、逆时针的 `(E,3)` 整数数组。`fixed` 是 `(N,2)` 布尔数组，
约束对应水平位移/速度分量。矩形生成器默认施加四周无穿透固壁。
第一版支持坐标分量约束；任意斜墙法向约束、开放/吸收边界尚未实现。
网格应覆盖初始湿域，积分点初始水深必须严格为正；岸线节点可以为零。

## 时间推进与异常

`step()` 采用固定步长，输入和输出状态必须分开。检测到非有限数值、单元翻转
或地形越界时抛出 `NumericalError`，输出和时间保持原样。
用户指定的 `dt` 不保证稳定，默认 `min_j=1e-8` 只是一道失效防护。

```python
solver.advance(state, duration=0.1, max_dt=0.001, adaptive=True)
```

`advance()` 分子步推进，拒绝时最多二分 12 次。若最终失败，`state` 保留最后
一个**已接受的子步**，不是整段 `duration` 开始时的状态。该机制不能替代
误差控制、激波处理或重网格。`model.suggested_dt()` 提供几何/CFL 初步估计，
不能保证移动网格在任意时长内不畸变。

## 验证与性能

```sh
.venv/bin/python -m pytest -q
.venv/bin/python validation/run_validation.py --device cpu --output validation/cpu_results.json
.venv/bin/python validation/run_validation.py --device cuda --scale --output validation/cuda_results.json
```

验收包含 MATLAB 10 秒对照、平底静水、线性小振幅驻波、抛物面水盆解析晃荡、
时间/空间收敛、CPU/CUDA 一致性及失败事务性。公式与适用范围见
[算例说明](docs/BENCHMARKS.md)，最新实测和限制见 [验证报告](validation/REPORT.md)。

绕岛三级细化为 32,544 节点、64,384 三角形单元，覆盖约 5 万单元目标量级。
岸线节点重新投影到初始零水深等高线；几何随加密变化，不能据此直接解释空间阶。
**加密绕岛默认仅运行 3 秒**。三级网格在更长时间出现强压缩，原步长、半步长和
四分之一步长分别在约 3.96、4.03、4.08 秒拒绝推进；尚未解决其 10 秒稳定性。
原始网格仍默认运行 10 秒。指定更长的 `--duration` 不代表该配置已验证。
默认晃荡三个周期的材料位移仍有约 6.6% 振幅的 RMS 误差，不能用守恒良好
替代形状精度判断。

当前不提供独立竖向速度、色散、溃坝激波正则化、任意湿干拓扑变化、自动重网格、
刚体耦合或自动微分。数学方法参考
[Wu & Zhong (2017), CMAME 318, 431–455](https://doi.org/10.1016/j.cma.2017.01.040)。

## delltower 远程查看

独立验证副本位于 delltower 的 `~/swe_engine_validation`，不改动原硬盘资料。
若在该副本启动浏览器服务：

```sh
cd ~/swe_engine_validation
PYTHONPATH=src .venv/bin/python -m swe --case island --device cuda --viewer viser --paused --hold
```

该验证副本已安装可选的 `viser` 依赖。新环境安装时使用 `pip install -e '.[cuda,web]'`。
然后在 Mac 建立隧道：

```sh
ssh -N -L 8080:127.0.0.1:8080 delltower
```

打开本地 `http://127.0.0.1:8080`。服务默认仅监听环回地址；查看器可完全关闭，
服务器无需桌面环境。当前远程目录为测试副本，本地 Git 仓库是开发源。
