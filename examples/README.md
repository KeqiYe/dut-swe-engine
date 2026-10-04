# 将 SWE 作为 Python 引擎使用

安装名是 `dut-swe-engine`，Python 导入名是 `swe`。这些示例位于引擎包外，
用公开接口定义场景、推进模拟和读取结果，可以复制到其他项目中运行。
不需要修改引擎源码，也不需要设置 `PYTHONPATH`。

| 示例 | 适合先看什么 |
| --- | --- |
| [basic.py](basic.py) | 最小 `import swe` → 建模 → `solver.step()` → 读取结果；自己控制时间循环 |
| [standing_wave.py](standing_wave.py) | 矩形水槽、平底和用户给定的初始水面函数 |
| [sloshing.py](sloshing.py) | 自定义三角形网格、抛物面水底、倾斜初始水面和可选动画 |
| [island.py](island.py) | 基础绕岛网格、高斯岛屿和入射脉冲，1006 单元 |
| [island_refined.py](island_refined.py) | 绕岛网格按单元总数加密，可关闭节点、显示三角形线框 |

`src/swe/cases.py` 保留内置基准用于 CLI 和数值回归。这里的示例把相应场景
定义展开在应用中，不调用 `swe.cases` 或 `swe.cli`。绕岛示例通过标准
`importlib.resources` 读取随安装包分发的基础网格，不依赖当前工作目录。

## 安装和运行

按[主 README](../README.md#快速开始) 下载项目、建立并激活虚拟环境，然后在项目根目录安装（开发本引擎时用 `-e`）：

```sh
python -m pip install -e .          # CPU 只需要 NumPy
python -m pip install -e '.[web]'   # 如需网页动画
```

安装后运行：

```sh
python examples/basic.py
python examples/standing_wave.py --device cpu
python examples/sloshing.py --device cpu
python examples/island.py --device cpu
```

无界面示例运行完自动退出。动画需显式开启：

```sh
python examples/sloshing.py --device cpu --viewer viser --speed 0.5 --loop --vertical-scale 5
```

打开 `http://127.0.0.1:8080`。`Stop server` 或终端 Ctrl+C 关闭；
不加 `--loop` 时运行一个周期后退出，加 `--hold` 则在终点停留。
`--vertical-scale` 只放大显示高度；`--speed` 只控制播放速度；
`--dt` 是真正的物理积分步长。可用 `--duration 0.1` 做短程检查。

切换 CUDA 只需 `--device cuda`；需要安装引擎的 `cuda` 可选依赖和 NVIDIA
GPU。默认 `auto` 在 CUDA 不可用时使用 CPU。示例所用数值模型仍是非色散、
深度平均的位移浅水方程，模型适用范围和稳定性限制见主 README。

## 绕岛网格加密和线框显示

```sh
python examples/island_refined.py --device cpu --mesh-factor 10 --viewer viser --hide-nodes --wireframe --loop --speed 0.2 --vertical-scale 5 --duration 4
```

`--mesh-factor 10` 将基础网格的 1006 个三角形增加到 **10,060 个**，倍率指单元总数。
采用均匀四分细化，再成对二分域内长边，保持网格连续；岸线投影到初始零水深
等高线，并保持均匀细化后的边界分辨率。因此这不是每条边等比例细分。
10 倍加密后默认时间步为 0.0015625 秒，默认模拟 0–3 秒；此配置已通过 CPU
0–4.5 秒检查，上述命令播放 0–4 秒。检查结果见
[验证记录](../validation/island_tenfold_cpu.json)。这不代表任意长时稳定，当前方法
在岸边强压缩时仍可能发生单元畸变。

`--hide-nodes` 关闭浏览器中的节点标记，`--wireframe` 只画水面三角形边线。
可随时在页面切换 **Show nodes** 和 **Wireframe**。面板显示当前顶点、单元数量。
竖向放大只用于显示，和网格加密倍率分别控制。

## 更高、更宽的入射波

```sh
python examples/island_refined.py --device cpu --mesh-factor 10 --amplitude 0.064 --width-scale 5 --dt 0.00078125 --duration 8 --viewer viser --hide-nodes --wireframe --loop --speed 0.1 --vertical-scale 5
```

这组参数已完成 8 秒 CPU 检查：物理波峰高于静水面 6.4 cm，为默认振幅的两倍，
静水位仍为 0.32 m；显示高度倍率仍为 5。每轮约 80 秒，实际速度受 CPU 算力限制。

`--amplitude` 单位为米，设置初始波峰相对静水面的高度。
`--width-scale` 控制同一振幅下脉冲的宽度，不改变峰值高度、峰值速度或波峰位置。
由于基础波宽也随振幅变化，上述参数的实际波宽约为最初 3.2 cm 波的 **3.69 倍**。
改变波形后会重新匹配初始岸线。波形和流速仍由初值驱动数值求解，并非播放解析动画。

直接将默认脉冲加高到 6.4 cm 会在约 3.86 秒出现单元翻转；加宽后选用上述配置，
这次仅验证到 8 秒。记录见 [八秒验证](../validation/island_broader_wave_5x.json)
和 [未加宽的高波测试](../validation/island_higher_wave.json)。

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

在其他项目使用时，安装引擎 wheel 或 `python -m pip install /path/to/dut-swe-engine`，
复制所需示例即可。源码发行包也包含本目录，wheel 中包含引擎和绕岛网格资源。
