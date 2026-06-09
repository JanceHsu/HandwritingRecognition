QT += widgets multimedia

TEMPLATE = app
TARGET = handwriting_recog

CONFIG -= debug debug_and_release
CONFIG += release c++17 warn_on force_debug_info

INCLUDEPATH += $$PWD/src

# Allow environment variable override (consistent with CMake Torch_DIR)
LIBTORCH_DIR_ENV = $$(LIBTORCH_DIR)
!isEmpty(LIBTORCH_DIR_ENV) {
    LIBTORCH_DIR = $$LIBTORCH_DIR_ENV
}
isEmpty(LIBTORCH_DIR) {
    LIBTORCH_DIR = D:/Develop/libtorch
}

INCLUDEPATH += $$LIBTORCH_DIR/include
INCLUDEPATH += $$LIBTORCH_DIR/include/torch/csrc/api/include

SOURCES += \
    src/main.cpp \
    src/airwritecontroller.cpp \
    src/mainwindow.cpp \
    src/canvas.cpp \
    src/recognizer.cpp

HEADERS += \
    src/airwritecontroller.h \
    src/mainwindow.h \
    src/canvas.h \
    src/recognizer.h

DEFINES += HANDWRITING_RECOG_DEFAULT_DEVICE=auto

win32 {
    LIBS += -lmf -lmfplat -lmfreadwrite -lmfuuid -lole32 -lshlwapi
    LIBS += -L$$LIBTORCH_DIR/lib -ltorch -ltorch_cpu -lc10

    exists($$LIBTORCH_DIR/lib/torch_cuda.lib) {
        LIBS += -lc10_cuda -ltorch_cuda -lcaffe2_nvrtc -lkineto

        # Link CUDA Toolkit runtime (required for torch::cuda::is_available)
        CUDA_PATH = $$(CUDA_PATH)
        isEmpty(CUDA_PATH) {
            CUDA_PATH = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8
        }
        exists($$CUDA_PATH/lib/x64/cudart.lib) {
            LIBS += $$quote(-L"$$CUDA_PATH/lib/x64") -lcudart
            INCLUDEPATH += $$quote("$$CUDA_PATH/include")
        }

        # Link NVTX (used by libtorch profiling)
        NVTX_PATH = C:/Program Files/NVIDIA Corporation/NvToolsExt
        exists($$NVTX_PATH/lib/x64/nvToolsExt64_1.lib) {
            LIBS += $$quote(-L"$$NVTX_PATH/lib/x64") -lnvToolsExt64_1
            INCLUDEPATH += $$quote("$$NVTX_PATH/include")
        }
    }

    QMAKE_POST_LINK += powershell -NoProfile -ExecutionPolicy Bypass -File $$PWD/scripts/deploy_qt_creator_build.ps1 -ProjectDir $$PWD -BuildRoot $$OUT_PWD -TargetConfig release -TargetName $$TARGET -LibtorchDir $$LIBTORCH_DIR
}