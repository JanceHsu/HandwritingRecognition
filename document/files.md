# 项目文件与目录清单

---

## 1. 源码

```
src/
├── main.cpp                    # 程序入口，初始化 QApplication 和 MainWindow
├── mainwindow.h/cpp            # 主窗口：UI 布局、摄像头预览、设备切换、识别调度、隔空书写
├── canvas.h/cpp                # 画布控件：鼠标绘制、程序化绘制（beginStroke/appendStroke/endStroke）
├── recognizer.h/cpp            # 识别器：LibTorch 模型加载、QImage 预处理、推理、设备管理
└── airwritecontroller.h/cpp    # 隔空书写控制器：Python 进程管理、帧发送、JSON 解析
```

---

## 2. 构建文件

| 文件 | 用途 |
|------|------|
| `CMakeLists.txt` | CMake 构建脚本，配置 Qt6、LibTorch、CUDA 依赖 |
| `HandwritingRecognition.pro` | qmake 构建脚本，供 Qt Creator 使用，含 CUDA 链接器标志和自动部署 |

---

## 3. 脚本

```
scripts/
├── train_mnist.py              # MNIST 模型训练与 TorchScript 导出
├── hand_tracker_service.py     # 隔空书写手部追踪服务（MediaPipe HandLandmarker）
├── export_test_images.py       # 导出 MNIST 测试图片供 C++ 推理验证
├── package_release.ps1         # CMake Release 构建打包（windeployqt + DLL 复制 + 启动脚本生成）
├── deploy_qt_creator_build.ps1 # Qt Creator 构建自动部署（QMAKE_POST_LINK 调用）
├── clean_project.ps1           # 清理 build/、dist/、Python 缓存
└── prepare_libtorch_cuda.ps1   # CUDA 版 LibTorch 环境预检
```

详见 [python.md](python.md)（Python 脚本）和 [powershell.md](powershell.md)（PowerShell 脚本）。

---

## 4. 文档

| 文档 | 内容 |
|------|------|
| [build.md](build.md) | 构建与运行：CMake 构建、Qt Creator 构建、差异对比、打包发布、常见问题 |
| [plan.md](plan.md) | 分阶段开发指导书：完整设计文档，覆盖环境准备到打包发布全流程 |
| [model.md](model.md) | 模型训练、推理与设备管理：CNN 结构、训练策略、预处理、CPU/CUDA 切换、预热 |
| [report.md](report.md) | 项目报告：背景、目标、方案、详细实现、实验结果、问题与解决、总结 |
| [innovation.md](innovation.md) | 项目创新点：多模态交互、CUDA 加速、可视化反馈、工程化架构等 |
| [python.md](python.md) | Python 脚本说明：train_mnist.py、hand_tracker_service.py、export_test_images.py |
| [airwriting.md](airwriting.md) | 隔空书写架构：Qt-Python 通信、手势判定、食指锁定、稳定策略 |
| [powershell.md](powershell.md) | PowerShell 脚本说明：package_release.ps1、deploy_qt_creator_build.ps1 等 |
| [files.md](files.md) | 文件清单（本文件） |
| [environment.txt](environment.txt) | 环境版本快照（Python、Qt、CUDA、LibTorch 版本） |
| [device.md](device.md) | 推理设备说明（已合并至 model.md） |

---

## 5. 模型与产物（构建时生成，不入版本控制）

| 目录 | 内容 |
|------|------|
| `artifacts/models/` | 训练产物：`mnist_model.pt`、`model.pth` |
| `build/` | CMake/qmake 构建输出（中间文件、可执行文件、DLL） |
| `dist/` | 发布包输出（`handwriting_recog.exe`、Qt/LibTorch DLL、模型、启动脚本） |

---

## 6. 根目录其他文件

| 文件 | 用途 |
|------|------|
| `run_handwriting_recog.bat` | 一键启动脚本，自动检查打包状态、设置 PATH、启动程序 |
| `README.md` | 项目说明（中文） |
| `.gitignore` | 版本控制忽略规则 |
