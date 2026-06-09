# Handwriting Recognition

Handwriting Recognition desktop project built with Qt + LibTorch.

Current status:
- Python training script scaffolded in `scripts/train_mnist.py`
- TorchScript export path implemented
- Qt Widgets UI and C++ inference layer scaffolded in `src/`
- `environment.txt` records the detected local versions

## Project layout

- `src/`: Qt Widgets application and LibTorch inference code.
- `scripts/`: training, packaging, cleanup, and CUDA preparation helpers.
- `artifacts/`: generated CPU/GPU model artifacts.
- `dist/`: packaged release output.
- `document/`: build and operation documentation.

Detailed build flow: [document/build_workflow.md](document/build_workflow.md)
Air-writing extension notes: [document/airwriting_extension.md](document/airwriting_extension.md)

## Build and Run on Windows (MSVC only)

Configure and build with MSVC 2022 and the Qt 6.11.1 MSVC toolchain:

```powershell
cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && cmake -S D:\Develop\Project\Qt\HandwritingRecognition -B D:\Develop\Project\Qt\HandwritingRecognition\build -G \"Visual Studio 17 2022\" -A x64 -DCMAKE_PREFIX_PATH=\"D:/Develop/Qt/6.11.1/msvc2022_64;D:/Develop/libtorch\" -DTorch_DIR=\"D:/Develop/libtorch/share/cmake/Torch\" && cmake --build D:\Develop\Project\Qt\HandwritingRecognition\build --config Release"
```

Qt Creator can open the repository directly as a CMake project. Use [CMakePresets.json](CMakePresets.json) and select the `qtcreator-msvc2022-release` configure preset when importing the folder.

If you want the simplest import path in Qt Creator, open [HandwritingRecognition.pro](HandwritingRecognition.pro) instead of the CMake file.

Run the executable with the MSVC Qt runtime first on PATH:

```powershell
$env:PATH = "D:\Develop\Qt\6.11.1\msvc2022_64\bin;D:\Develop\libtorch\lib;$env:PATH"
D:\Develop\Project\Qt\HandwritingRecognition\build\Release\handwriting_recog.exe
```

Package a distributable folder:

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\package_release.ps1
```

The packaged release folder is the practical standalone delivery format for this project. It is not a single-file binary; it is a self-contained directory with `handwriting_recog.exe`, Qt runtime DLLs, LibTorch DLLs, and the model files under `models/`. See [document/build_workflow.md](document/build_workflow.md) for the packaging order and required contents.

`build/` is the local development output, while `dist/` is the packaged snapshot used for distribution and one-click launch.

Current air-writing runtime:

- The tracker runs as a Python 3.13 process using OpenCV + MediaPipe Tasks.
- The Qt app talks to `scripts/hand_tracker_service.py` through `QProcess` and JSON lines.
- No Python virtual environment is required for this workspace.
- The hand tracker model is downloaded automatically on first run if it is missing.

Prepare switch to LibTorch 2.5.1 CUDA:

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\prepare_libtorch_cuda.ps1 -CudaLibtorchDir D:\Develop\libtorch-cuda
```

Runtime override example for CUDA package/build:

```powershell
$env:LIBTORCH_DIR = "D:\Develop\libtorch-cuda"
$env:LIBTORCH_DEVICE = "cuda"
D:\Develop\Project\Qt\HandwritingRecognition\run_handwriting_recog.bat
```

The log field `推理设备=cpu` means the current runtime is doing inference on CPU. It does not mean the model was trained on CPU; the training script already runs on CUDA when you pass `--device cuda` or use the default auto mode on a CUDA-capable machine.

The window now includes a model selector with `CPU 模型` and `GPU 模型（推荐）`. The GPU model option is only selectable when CUDA is available.

If the app crashes on the first recognition click, use the new QImage-based recognition path and check the logs in the sidebar before retrying. The app now warms the model at startup and blocks re-entry while a recognition request is active.

CUDA training recommendation:

```powershell
py -3.13 scripts\train_mnist.py --epochs 50 --batch-size 256 --output-dir artifacts --device cuda --pin-memory --num-workers 2
```

One-click launch from Explorer:

```text
D:\Develop\Project\Qt\HandwritingRecognition\run_handwriting_recog.bat
```

Project cleanup (remove generated build/package/cache directories):

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\clean_project.ps1
```