# 项目文件与目录清单

本文件为工作区当前文件/目录的逐条说明、用途、是否为最终产物，以及推荐的保留/删除操作。按项目主线（源码、模型、打包、脚本、文档、临时产物）分组。

---

## 1. 源码（必须保留）

- `src/`：主程序源码，包含 Qt UI、Canvas、识别器实现（`recognizer.cpp/h`）、本地 OpenCV shim（已停用识别路径）等。
  - 保留理由：项目核心逻辑。

- `CMakeLists.txt`：CMake 构建脚本，MSVC/Qt + LibTorch 配置。
  - 保留理由：构建入口。

- `run_handwriting_recog.bat`：一键本地运行脚本（会使用 `dist` 中的可执行并设置环境变量）。
  - 保留理由：便捷运行。可按需修改。

- `scripts/`：包含训练、打包、清理脚本。
  - `train_mnist.py`：训练与导出模型脚本（Python）。
  - `package_release.ps1`：Windows 打包脚本（windeployqt + 复制模型/LibTorch DLL）。
  - `clean_project.ps1`：清理脚本（删除 `build/`, `dist/`, 可选数据缓存）。
  - `prepare_libtorch_cuda.ps1`：LibTorch CUDA 切换/准备脚本。
  - 保留理由：项目维护与发布自动化。

- `document/`：项目说明与构建文档。主要文件：
  - `build_workflow.md`：新的构建全流程（已存在）。
  - `run.md`、`document.md`：历史分阶段说明与设计记录（保留为历史文档）。
  - `file_inventory.md`（本文件）
  - 保留理由：文档记录与操作指南。

## 2. 模型与产物（保留分类模型）

- `artifacts/`：训练脚本输出目录（应包含 `models/cpu` 与 `models/gpu` 子目录，各含 `mnist_model.pt` 与 `model.pth`）。
  - 保留理由：训练产物，必要用于打包与运行验证。

- `dist/`：发布包输出目录（包含 `handwriting_recog.exe`、Qt/LibTorch 运行时库、`models/` 子目录）。
  - 保留理由：发布产物，可供直接分发/测试。

## 3. 生成/临时目录（推荐删除 | 可恢复）

- `build/`：CMake/MSBuild 的构建输出目录（包含临时编译文件、生成的 MOC 源、可执行）。
  - 建议：删除（由 `clean_project.ps1` 可重建）。
  - 风险：删除后需要重新构建来产生可执行。

- `test_images/`：用于导出或测试的图片样例目录（可占磁盘）。
  - 建议：若不需要保留测试样例，可删除或迁移到 `artifacts/test_images`。

- `__pycache__/`, `*.pyc`, `*.pyo`：Python 缓存文件。
  - 建议：删除。

- `artifacts_gpu_check/`, `artifacts_gpu_check2/`：临时实验目录（若存在）。
  - 建议：删除。

## 4. 日志 / 报告（可选保留/删除）

- `test_log.md`：测试日志（临时）。
  - 建议：如无长期价值可删除；若用于复现问题则保留。

- `experiment_report.md`：训练/实验报告。
  - 建议：保留，便于回溯训练参数与结果。

## 5. 配置 / 依赖清单（保留）

- `requirements.txt`：Python 依赖（保留）。
- `environment.txt`：记录的本地环境（保留）。

## 6. .gitignore（已更新）

- 当前 `.gitignore` 包含：`build/`, `dist/`, `artifacts/` 等临时文件模式。请确保在提交前 `.gitignore` 覆盖所有生成产物，避免历史垃圾文件进入仓库。

---

## 推荐的清理动作（已执行或将执行）

1. 删除：`build/`（构建产物）
2. 删除：`test_log.md`（临时日志）
3. 删除：临时实验目录 `artifacts_gpu_check*`（若存在）
4. 保留：`dist/`、`artifacts/models/*`、`src/`、`scripts/`、`document/`、`README.md`、`requirements.txt` 等

---

如果你同意，我将执行第 1、2、3 步（删除 `build/`、`test_log.md`、临时实验目录），并在完成后更新任务清单与 `.gitignore` 状态。