QT += widgets multimedia

TEMPLATE = app
TARGET = handwriting_recog

CONFIG += c++17 warn_on

INCLUDEPATH += $$PWD/src

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
        LIBS += -lc10_cuda -ltorch_cuda -lcaffe2_nvrtc
    }

    CONFIG(debug, debug|release) {
        QMAKE_POST_LINK += powershell -NoProfile -ExecutionPolicy Bypass -File $$PWD/scripts/deploy_qt_creator_build.ps1 -ProjectDir $$PWD -BuildRoot $$OUT_PWD -TargetConfig debug -TargetName $$TARGET -LibtorchDir $$LIBTORCH_DIR
    } else {
        QMAKE_POST_LINK += powershell -NoProfile -ExecutionPolicy Bypass -File $$PWD/scripts/deploy_qt_creator_build.ps1 -ProjectDir $$PWD -BuildRoot $$OUT_PWD -TargetConfig release -TargetName $$TARGET -LibtorchDir $$LIBTORCH_DIR
    }
}