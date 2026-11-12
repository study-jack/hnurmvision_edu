# 跃鹿战队算法组二阶段培训与考核

本仓库是面向学员的自瞄教学骨架。

## 目录结构

```text
hnurmvision-training/
├── CMakeLists.txt              # 培训主程序构建配置
├── README.md
├── configs/
│   └── simulation.yaml         # 显示、发射许可、相机内参和坐标变换
├── include/
│   ├── lessons.hpp             # 第二至第五讲函数声明
│   ├── tracking_state.hpp      # 第四讲跨帧状态
│   └── types.hpp               # 各讲之间传递的数据类型
├── io/
│   ├── config.cpp/.hpp         # YAML 读取和校验
│   └── io_context.hpp          # 第二讲 IO 状态与四个操作声明
├── lessons/
│   ├── lesson02_io.cpp         # 第二讲：仿真 IO
│   ├── lesson03_detection.cpp  # 第三讲：装甲板检测
│   ├── lesson04_tracking.cpp   # 第四讲：位姿与跟踪
│   └── lesson05_aiming.cpp     # 第五讲：弹道与瞄准
├── src/
│   └── main.cpp                # 单线程顺序主循环
├── scripts/                    # 构建和启动入口
├── docs/                       # 课程讲义和仿真环境简介
└── simulation/
    ├── simulator/              # Bevy 仿真器源码、资产和 config.toml
    ├── ros2/                   # 仿真通信使用的 hnurm_interfaces
    └── tools/                  # 场地生成辅助工具
```

## 编译

### 环境要求

- Ubuntu 22.04
- ROS2 Humble、colcon、ament_cmake
- CMake、支持 C++17 的编译器
- OpenCV、Eigen3、yaml-cpp
- Rustup 和 Rust 1.95.0（只在编译仿真器时需要）

所有命令都在仓库根目录执行。首次使用时先编译主程序及 ROS2 消息，再编译仿真器。

### 1. 编译培训主程序

```bash
./scripts/build_training.sh
```

该脚本依次完成：

1. 编译 `simulation/ros2/hnurm_interfaces`；
2. 配置根目录 CMake 工程；
3. 生成 `build/training`。

### 2. 编译仿真器

```bash
./scripts/build_simulator.sh
```

该脚本会生成教学场地、以 ROS2 特性编译 Bevy 仿真器，并生成：

```text
simulation/simulator/bin/daedalus
```

首次编译时间较长。修改仿真器 Rust 源码、资产生成逻辑或 ROS2 接口后需要重新执行；只修改
`lessons/`、`io/` 或 `src/` 时，重新执行 `build_training.sh`。

## 运行指南

运行时需要两个终端，并且两个终端都位于仓库根目录。

### 终端 1：启动仿真器

```bash
./scripts/run_simulator.sh
```

看到仿真窗口和训练场地后，再启动培训程序。脚本会自动配置 ROS2 环境、
`ROS_DOMAIN_ID=89`、动态库路径和仿真资产路径。

### 终端 2：启动培训程序

```bash
./scripts/run_training.sh
```

该脚本使用 `configs/simulation.yaml` 启动 `build/training`。初始骨架等待输入，按 Ctrl+C 退出；此时没有图像窗口和 ROS2 通信。


### 退出

- 完成第二讲并显示图像后，在培训图像窗口按 `q` 或 `Esc`；
- 或在对应终端按 `Ctrl+C`；
- 建议先退出培训程序，再关闭仿真器。

### 常用配置

运行配置位于 `configs/simulation.yaml`；学员还需要完成 `lessons/` 和 IO 骨架：

- `display`：是否显示培训图像窗口；
- `allow_fire`：是否允许最终发射，默认 `false`；
- 相机内参、畸变参数和相机到云台的坐标变换。

仿真场景和靶车由 `simulation/simulator/config.toml` 配置。`[training] enabled=true` 时生成
教学观察车和训练靶，设为 `false` 时回到手动场景。

## ONNX Runtime 依赖（已随仓库提供）

`third_party/onnxruntime/` 内置 CPU 版 ONNX Runtime 1.23.2，无需另外安装 SDK：

```text
third_party/onnxruntime/
├── include/                        # C/C++ API 头文件
└── lib/
    ├── libonnxruntime.so.1.23.2    # CPU 推理库
    ├── libonnxruntime.so.1 -> libonnxruntime.so.1.23.2
    └── libonnxruntime.so -> libonnxruntime.so.1
```

`CMakeLists.txt` 用 `${ORT_ROOT}` 指向该目录并设置 `BUILD_RPATH`，编译产物可直接运行。
若要用 GPU 推理：把 `lib/` 换成 onnxruntime-gpu 的同名库，并把
`libonnxruntime_providers_cuda.so`、cuDNN/cuBLAS 运行库放进同一目录即可，代码与模型无需修改。

模型默认读取 `assets/armor_pose.onnx`，可用环境变量 `ARMOR_ONNX` 指向其他模型；
设置 `ARMOR_DETECTOR=opencv` 可强制使用 OpenCV 兜底检测，便于对比效果。

## 换机器复现步骤

1. 按上文「编译 → 1. 编译培训主程序」执行 `./scripts/build_training.sh`；
2. 需要看仿真画面时再执行 `./scripts/build_simulator.sh`（耗时较长）；
3. 两个终端分别运行 `./scripts/run_simulator.sh` 与 `./scripts/run_training.sh`；
4. 培训窗口里绿色框为大装甲板、橙色框为小装甲板，标签为类别（B1–B5/BO/BS/R1–R5/RO/RS）与置信度。
