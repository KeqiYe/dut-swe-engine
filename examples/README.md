# 将 SWE 作为 Python 引擎使用

安装名是 `dut-swe-engine`，Python 导入名是 `swe`。这些示例位于引擎包外，
用公开接口定义场景、推进模拟和读取结果，可以复制到其他项目中运行。
不需要修改引擎源码，也不需要设置 `PYTHONPATH`。

| 示例 | 适合先看什么 |
| --- | --- |
| [basic.py](basic.py) | 最小 `import swe` → 建模 → `solver.step()` → 读取结果；自己控制时间循环 |
| [standing_wave.py](standing_wave.py) | 矩形水槽、平底和用户给定的初始水面函数 |
| [sloshing.py](sloshing.py) | 自定义三角形网格、抛物面水底、倾斜初始水面和可选动画 |
| [island.py](island.py) | 原 MATLAB 绕岛网格、高斯岛屿和入射脉冲，原始 1006 单元 |

`src/swe/cases.py` 保留内置基准用于 CLI 和数值回归。这里的示例把相应场景
定义展开在应用中，不调用 `swe.cases` 或 `swe.cli`。绕岛示例通过标准
`importlib.resources` 读取随安装包分发的原始网格，不依赖当前工作目录。

## 安装和运行

在项目根目录安装一次（开发本引擎时用 `-e`）：

```sh
python -m pip install -e .          # CPU 只需要 NumPy
python -m pip install -e '.[web]'   # 如需网页动画
```

本项目现有虚拟环境已经安装好了，可以直接运行：

```sh
.venv/bin/python examples/basic.py
.venv/bin/python examples/standing_wave.py --device cpu
.venv/bin/python examples/sloshing.py --device cpu
.venv/bin/python examples/island.py --device cpu
```

无界面示例运行完自动退出。动画需显式开启：

```sh
.venv/bin/python examples/sloshing.py --device cpu --viewer viser --speed 0.5 --loop --vertical-scale 5
```

打开 `http://127.0.0.1:8080`。`Stop server` 或终端 Ctrl+C 关闭；
不加 `--loop` 时运行一个周期后退出，加 `--hold` 则在终点停留。
`--vertical-scale` 只放大显示高度；`--speed` 只控制播放速度；
`--dt` 是真正的物理积分步长。可用 `--duration 0.1` 做短程检查。

切换 CUDA 只需 `--device cuda`；需要安装引擎的 `cuda` 可选依赖和 NVIDIA
GPU。默认 `auto` 在 CUDA 不可用时使用 CPU。示例所用数值模型仍是非色散、
深度平均的位移浅水方程，模型适用范围和稳定性限制见主 README。

## 应用与引擎各自负责什么

应用负责：重力、网格、地形、水面与速度初值、设备、时间步长、模拟时长。
修改各示例的 `build_model()` 参数或函数体即可定义自己的问题。

引擎负责：`ModelBuilder` 预处理、`Model`/`State` 数据、
`DisplacementSolver` 的 CPU/CUDA 数值推进。需要完全控制循环时：

```python
import swe

# model 由应用用 swe.ModelBuilder 构建。
state, next_state = model.state(), model.state()
solver = swe.DisplacementSolver(model)
for _ in range(100):
    solver.step(state, next_state, dt=0.005)
    state, next_state = next_state, state
snapshot = state.to_numpy()
```

也可以调用 `state = swe.run(model, dt=0.005, duration=1.0)`。
它从模型初态运行到给定时长，返回最终 `State`；`viewer=None` 时无显示依赖，
也不限制计算速度。若需从已有状态续算，用 `solver.step()` 或 `solver.advance()`。

动画示例向 `swe.run()` 传入由 `swe.viewers.create_viewer()` 创建的可选查看器，
并在 `finally` 中关闭它。引擎运行入口接收任何应用构建的模型，不绑定内置算例名。
循环播放每次都从初态重新计算，并不是把多个周期连续积分。

在其他项目使用时，安装引擎 wheel 或 `python -m pip install /path/to/swe_engine`，
复制所需示例即可。源码发行包也包含本目录，wheel 中包含引擎和绕岛网格资源。
