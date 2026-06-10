# Qt Creator 与命令行 CMake 构建差异

本文记录本项目在两种构建入口下的差异：

- Qt Creator 中通过 [HandwritingRecognition.pro](../HandwritingRecognition.pro) 导入并构建
- 命令行中通过 [CMakeLists.txt](../CMakeLists.txt) + `cmake -S/-B` 构建

这份说明以本仓库当前状态为准，重点解决两个问题：

1. Qt Creator 构建后无法稳定使用 CUDA 推理
2. 两种构建方式在调试、发布、部署和运行环境上的差异不清晰

## 一句话结论

- 命令行 CMake 构建更适合“可复现、可打包”的发布流程
- Qt Creator qmake 构建更适合“快速打开、直接调试、看 UI 状态”的开发流程
- 两者编译的是同一套源码，但默认运行环境和产物布局不同，所以表现可能不同

## 入口文件对比

| 维度 | Qt Creator | 命令行 CMake |
|---|---|---|
| 导入文件 | [HandwritingRecognition.pro](../HandwritingRecognition.pro) | [CMakeLists.txt](../CMakeLists.txt) |
| 适用场景 | IDE 内直接打开、看界面、点按钮调试 | 命令行构建、自动化、打包、CI |
| 默认工具链 | Qt Creator 选择的 Kit | `cmake` 命令显式指定生成器和路径 |
| 依赖部署 | 依赖 qmake post-link 部署脚本 [scripts/deploy_qt_creator_build.ps1](../scripts/deploy_qt_creator_build.ps1) | 依赖 `scripts/package_release.ps1` |

## 构建流程对比

### Qt Creator qmake

1. 打开 `HandwritingRecognition.pro`
2. 选择 Qt 6.11.1 / MSVC 2022 Kit
3. Qt Creator 生成 Debug 或 Release 构建目录
4. 构建完成后执行 qmake 的 post-link 部署动作 [scripts/deploy_qt_creator_build.ps1](../scripts/deploy_qt_creator_build.ps1)
5. 运行时从构建目录直接加载模型和 LibTorch DLL

### 命令行 CMake

1. 用 Visual Studio / vcvars 初始化 MSVC 环境
2. 运行 `cmake -S ... -B ... -G "Visual Studio 17 2022" -A x64`
3. 再运行 `cmake --build ... --config Release`
4. 通过 `scripts/package_release.ps1` 复制 Qt、LibTorch 和模型文件到 `dist`
5. 从 `dist` 或 `run_handwriting_recog.bat` 启动

## Debug 与 Release 的区别

### Debug

- 带调试符号，适合断点、变量查看、调用栈分析
- 优化较少，运行速度慢一点，但更利于定位问题
- 产物通常放在 `debug/` 或对应的 Debug 输出目录
- Qt Creator 中更容易看到界面行为和日志输出

### Release

- 开启优化，运行速度更快
- 适合接近真实用户环境的验证
- 产物通常放在 `release/` 或对应的 Release 输出目录
- 命令行 CMake 的发布验证通常以 Release 为准

### 实际影响

- Debug 里能跑，不代表 Release 一定能跑，反过来也一样
- Qt Creator 的 Debug/Release 如果没有复制 LibTorch DLL 和模型文件，运行表现会和命令行打包版不一致
- 本项目的 GPU 识别依赖 CUDA 侧 LibTorch DLL 正确出现在运行时路径里

## 横向对比

| 项目 | Qt Creator qmake | 命令行 CMake |
|---|---|---|
| 构建入口 | `.pro` | `CMakeLists.txt` |
| 构建命令 | IDE 点击构建 | `cmake -S/-B` + `cmake --build` |
| 运行目录 | Qt Creator 的构建目录 | `build/Release` 或 `dist/` |
| 模型部署 | qmake post-link 复制到构建目录 | `package_release.ps1` 复制到 `dist/models` |
| Qt DLL 部署 | 依赖 Kit / 可选 `windeployqt` | `package_release.ps1` 调用 `windeployqt` |
| LibTorch DLL 部署 | qmake post-link 复制到构建目录 | `package_release.ps1` 复制到 `dist` |
| CUDA 识别 | 取决于运行目录是否能加载 CUDA 版 LibTorch | 取决于 `dist` 是否包含 CUDA 版 DLL |
| 适合人群 | 本地调试、UI 迭代 | 发布、脚本化构建、归档 |

## 纵向对比

### 1. 源码层

- 两种方式编译的是同一套 `src/` 源码
- 识别逻辑、摄像头逻辑、模型加载逻辑完全一致
- 差异主要来自构建系统如何设置 include/lib 路径和运行时部署

### 2. 编译层

- CMake 通过 `find_package(Qt6)` 和 `find_package(Torch)` 解析依赖
- qmake 通过 `INCLUDEPATH`、`LIBS` 和 `QMAKE_POST_LINK` 手工管理依赖
- CMake 更适合大项目依赖管理，qmake 更适合 Qt Creator 里快速上手

### 3. 运行层

- CMake 命令行构建通常配合 `run_handwriting_recog.bat` 或 `dist` 运行
- Qt Creator qmake 构建如果不额外部署，常常只是一堆 exe / obj 文件
- 本项目已经把 qmake 运行时补成“构建完可直接运行”的方式：复制模型和 LibTorch DLL

### 4. 部署层

- CMake 发布由 `scripts/package_release.ps1` 负责
- Qt Creator 发布由 qmake post-link 脚本 [scripts/deploy_qt_creator_build.ps1](../scripts/deploy_qt_creator_build.ps1) 负责
- 两者最终都应得到：Qt 运行库、LibTorch DLL、模型文件

## 为什么 Qt Creator 里会”识别不到 CUDA”

常见原因有三个：

1. 运行目录里只有 exe，没有 CUDA 版 LibTorch DLL，所以 `torch::cuda::is_available()` 返回 false
2. 运行目录里没有模型文件，`resolveModelPathWithFallback()` 找不到 `artifacts/models` 或 `models`
3. Qt Creator 使用了和命令行不同的 Kit / 环境变量，导致 `LIBTORCH_DIR` 不一致

本仓库当前的修复策略是：

- qmake 构建时链接 CUDA 版 LibTorch
- qmake 构建后自动复制 LibTorch DLL 和模型文件到构建目录
- 运行时代码向上搜索 `models` / `artifacts/models` / `dist/models`

## 推荐使用方式

- 如果你要改 UI、看日志、快速验证手势识别，优先用 Qt Creator + `HandwritingRecognition.pro`
- 如果你要做可复现发布、打包、交付，优先用命令行 CMake + `scripts/package_release.ps1`
- 如果 Qt Creator 和命令行行为不一致，优先检查运行目录里是否真的有 `torch_cuda.dll`、`c10_cuda.dll` 和 `models/`

## 最后建议

如果 Qt Creator 里仍然显示 `CUDA available: false`，优先检查这三项：

1. Qt Creator 的运行目录是否是当前构建目录
2. 构建目录里是否已经复制了 LibTorch CUDA DLL
3. Qt Creator 使用的 Kit 是否和 `D:/Develop/libtorch` 是同一套 MSVC 架构