#include "mainwindow.h"

#include "airwritecontroller.h"
#include "canvas.h"
#include "recognizer.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMessageBox>
#include <QCameraDevice>
#include <QCamera>
#include <QMediaCaptureSession>
#include <QMediaDevices>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QPainter>
#include <QPen>
#include <QColor>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QVideoFrame>
#include <QVideoSink>
#include <QStatusBar>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

#include <torch/cuda.h>

namespace {

bool hasVisibleInk(const QImage& img)
{
    if (img.isNull()) {
        return false;
    }

    const QImage gray = img.convertToFormat(QImage::Format_Grayscale8);
    if (gray.isNull()) {
        return false;
    }

    int inkPixels = 0;
    const int width = img.width();
    const int height = img.height();
    for (int row = 0; row < height; ++row) {
        const auto* line = gray.constScanLine(row);
        for (int col = 0; col < width; ++col) {
            if (static_cast<unsigned char>(line[col]) < 245) {
                ++inkPixels;
            }
        }
    }

    return inkPixels >= 20;
}

QString candidateModelPath(const QString& rootDir, const QString& modelKey)
{
    const QString candidate = QDir(rootDir).filePath(modelKey + "/mnist_model.pt");
    return QFileInfo::exists(candidate) ? QFileInfo(candidate).absoluteFilePath() : QString();
}

QString resolveModelPathWithFallback(const QString& modelKey)
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList roots = {
        QDir(appDir).filePath("models"),
        QDir(appDir).filePath("../models"),
        QDir(appDir).filePath("../../models"),
        QDir(appDir).filePath("../artifacts/models"),
        QDir(appDir).filePath("../../artifacts/models"),
        QDir(appDir).filePath("../dist/models"),
        QDir(appDir).filePath("../../dist/models"),
    };

    const QString fallbackKey = (modelKey == "gpu") ? QStringLiteral("cpu") : QStringLiteral("gpu");

    for (const QString& root : roots) {
        const QString candidate = candidateModelPath(root, modelKey);
        if (!candidate.isEmpty()) {
            return candidate;
        }
    }

    for (const QString& root : roots) {
        const QString candidate = candidateModelPath(root, fallbackKey);
        if (!candidate.isEmpty()) {
            return candidate;
        }
    }

    return candidateModelPath(QDir(appDir).filePath("models"), modelKey);
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    airController_ = new AirWriteController(this);
    buildUi();
    bindEvents();
    appendLog("Application started");

    appendLog(QString("CUDA available: %1").arg(torch::cuda::is_available() ? "true" : "false"));
    appendLog(QString("LIBTORCH_DEVICE=%1").arg(qEnvironmentVariable("LIBTORCH_DEVICE", "<unset>")));

    connect(airController_, &AirWriteController::trackingUpdated, this, &MainWindow::onAirTrackingUpdated);
    connect(airController_, &AirWriteController::trackingLost, this, &MainWindow::onAirTrackingLost);
    connect(airController_, &AirWriteController::statusMessage, this, &MainWindow::appendLog);

    refreshCameraList();
    loadRecognizerForSelection();
}

MainWindow::~MainWindow()
{
    stopCameraPreview();
    delete recognizer_;
}

void MainWindow::buildUi()
{
    setWindowTitle("Handwriting Recognition");
    resize(1100, 760);

    auto* centralWidget = new QWidget(this);
    centralWidget->setStyleSheet(
        "QWidget { background: #f5f1e8; }"
        "QFrame#card { background: white; border-radius: 18px; }"
        "QLabel#title { font-size: 24px; font-weight: 700; color: #1f1f1f; }"
        "QLabel#result { font-size: 22px; font-weight: 600; color: #212121; }"
        "QPushButton { padding: 10px 20px; border-radius: 10px; font-size: 15px; }"
        "QPushButton#primary { background: #1e88e5; color: white; }"
        "QPushButton#primary:hover { background: #1565c0; }"
        "QPushButton#secondary { background: #eceff1; color: #202124; }"
        "QPushButton#secondary:hover { background: #d7dde2; }"
    );

    auto* rootLayout = new QHBoxLayout(centralWidget);
    rootLayout->setContentsMargins(24, 24, 24, 24);
    rootLayout->setSpacing(20);

    auto* leftColumn = new QVBoxLayout();
    leftColumn->setSpacing(20);

    auto* cameraCard = new QFrame(centralWidget);
    cameraCard->setObjectName("card");
    auto* cameraLayout = new QVBoxLayout(cameraCard);
    cameraLayout->setContentsMargins(18, 18, 18, 18);
    cameraLayout->setSpacing(12);

    auto* cameraTitle = new QLabel("隔空书写预览", cameraCard);
    cameraTitle->setObjectName("title");
    cameraTitle->setStyleSheet("font-size: 18px; font-weight: 700; color: #1f1f1f;");

    cameraLabel_ = new QLabel("摄像头选择", cameraCard);
    cameraLabel_->setStyleSheet("font-size: 15px; font-weight: 600; color: #202124;");

    cameraComboBox_ = new QComboBox(cameraCard);
    cameraComboBox_->setEnabled(false);

    cameraPreviewLabel_ = new QLabel("摄像头未开启", cameraCard);
    cameraPreviewLabel_->setAlignment(Qt::AlignCenter);
    cameraPreviewLabel_->setMinimumSize(400, 240);
    cameraPreviewLabel_->setStyleSheet(
        "QLabel { background: #111827; color: #cbd5e1; border-radius: 12px; border: 1px solid #374151; font-size: 14px; }"
    );

    airModeButton_ = new QPushButton("开启隔空书写", cameraCard);
    airModeButton_->setCheckable(true);
    airModeButton_->setObjectName("primary");

    cameraLayout->addWidget(cameraTitle);
    cameraLayout->addWidget(cameraLabel_);
    cameraLayout->addWidget(cameraComboBox_);
    cameraLayout->addWidget(cameraPreviewLabel_, 1);
    cameraLayout->addWidget(airModeButton_);

    auto* canvasCard = new QFrame(centralWidget);
    canvasCard->setObjectName("card");
    auto* canvasLayout = new QVBoxLayout(canvasCard);
    canvasLayout->setContentsMargins(18, 18, 18, 18);

    auto* titleLabel = new QLabel("手写画布", canvasCard);
    titleLabel->setObjectName("title");
    canvas_ = new Canvas(canvasCard);
    canvas_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    canvasLayout->addWidget(titleLabel);
    canvasLayout->addWidget(canvas_, 1);

    leftColumn->addWidget(cameraCard, 1);
    leftColumn->addWidget(canvasCard, 2);

    auto* sideCard = new QFrame(centralWidget);
    sideCard->setObjectName("card");
    auto* sideLayout = new QVBoxLayout(sideCard);
    sideLayout->setContentsMargins(22, 22, 22, 22);
    sideLayout->setSpacing(16);

    resultLabel_ = new QLabel("识别结果：", sideCard);
    resultLabel_->setObjectName("result");
    resultLabel_->setWordWrap(true);

    modelLabel_ = new QLabel("模型选择", sideCard);
    modelLabel_->setStyleSheet("font-size: 15px; font-weight: 600; color: #202124;");

    modelComboBox_ = new QComboBox(sideCard);
    modelComboBox_->addItem("GPU 模型（推荐）", "gpu");
    modelComboBox_->addItem("CPU 模型", "cpu");
    if (!torch::cuda::is_available()) {
        modelComboBox_->removeItem(0);
        modelComboBox_->setCurrentIndex(0);
    } else {
        modelComboBox_->setCurrentIndex(0);
    }

    auto* airHintLabel = new QLabel("Python + OpenCV + MediaPipe 整手关键点追踪；食指指尖用于落笔，中指抬起时暂停绘制。", sideCard);
    airHintLabel->setWordWrap(true);
    airHintLabel->setStyleSheet("color: #5f6368; font-size: 13px;");

    mirrorPreviewCheckBox_ = new QCheckBox("镜像翻转摄像头画面", sideCard);
    mirrorPreviewCheckBox_->setChecked(false);
    mirrorPreviewCheckBox_->setStyleSheet("font-size: 13px; color: #202124;");

    auto* hintLabel = new QLabel("请在画布上书写 0 到 9 的数字，然后点击“识别”。", sideCard);
    hintLabel->setWordWrap(true);
    hintLabel->setStyleSheet("color: #5f6368; font-size: 14px;");

    recognizeButton_ = new QPushButton("识别", sideCard);
    recognizeButton_->setObjectName("primary");

    clearButton_ = new QPushButton("清空", sideCard);
    clearButton_->setObjectName("secondary");

    auto* logLabel = new QLabel("操作日志", sideCard);
    logLabel->setStyleSheet("font-size: 15px; font-weight: 600; color: #202124;");

    logView_ = new QPlainTextEdit(sideCard);
    logView_->setReadOnly(true);
    logView_->setMaximumBlockCount(300);
    logView_->setMinimumHeight(140);
    logView_->setStyleSheet(
        "QPlainTextEdit { background: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 8px; font-size: 13px; }"
    );

    sideLayout->addWidget(resultLabel_);
    sideLayout->addWidget(airHintLabel);
    sideLayout->addWidget(mirrorPreviewCheckBox_);
    sideLayout->addWidget(modelLabel_);
    sideLayout->addWidget(modelComboBox_);
    sideLayout->addWidget(hintLabel);
    sideLayout->addStretch(1);
    sideLayout->addWidget(recognizeButton_);
    sideLayout->addWidget(clearButton_);
    sideLayout->addWidget(logLabel);
    sideLayout->addWidget(logView_, 1);

    rootLayout->addLayout(leftColumn, 2);
    rootLayout->addWidget(sideCard, 1);

    setCentralWidget(centralWidget);

    auto* footer = new QLabel("Made by Jance", this);
    footer->setStyleSheet("color: #6b6b6b; font-size: 11px;");
    footer->setAlignment(Qt::AlignCenter);
    footer->setFixedHeight(18);
    statusBar()->addWidget(footer, 1);
    statusBar()->setSizeGripEnabled(false);
    statusBar()->show();
}

void MainWindow::bindEvents()
{
    connect(airModeButton_, &QPushButton::toggled, this, &MainWindow::onAirModeToggled);
    connect(cameraComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onCameraSelectionChanged);
    connect(mirrorPreviewCheckBox_, &QCheckBox::toggled, this, &MainWindow::onPreviewMirrorToggled);
    connect(recognizeButton_, &QPushButton::clicked, this, &MainWindow::onRecognize);
    connect(clearButton_, &QPushButton::clicked, this, &MainWindow::onClear);
    connect(modelComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onModelSelectionChanged);
}

void MainWindow::setRecognitionEnabled(bool enabled)
{
    recognizeButton_->setEnabled(enabled);
}

void MainWindow::setRecognitionBusy(bool busy)
{
    recognitionBusy_ = busy;
    recognizeButton_->setEnabled(!busy && recognizer_ != nullptr);
    clearButton_->setEnabled(!busy);
    if (modelComboBox_ != nullptr) {
        modelComboBox_->setEnabled(!busy);
    }
}

QString MainWindow::selectedModelKey() const
{
    if (modelComboBox_ == nullptr) {
        return "cpu";
    }
    return modelComboBox_->currentData().toString();
}

QString MainWindow::resolveModelPath(const QString& modelKey) const
{
    return resolveModelPathWithFallback(modelKey);
}

void MainWindow::loadRecognizerForSelection()
{
    const QString modelKey = selectedModelKey();
    const QString modelPath = resolveModelPath(modelKey);
    appendLog(QString("准备加载模型: key=%1, path=%2").arg(modelKey, modelPath));

    setRecognitionBusy(true);
    delete recognizer_;
    recognizer_ = nullptr;

    try {
        recognizer_ = new DigitRecognizer(modelPath.toStdString());
        try {
            recognizer_->warmUp();
            appendLog("模型预热完成");
        } catch (const std::exception& warmupError) {
            appendLog(QString("模型预热失败: %1").arg(warmupError.what()));
        }
        setRecognitionEnabled(true);
        resultLabel_->setText("识别结果：");
        appendLog(QString("模型加载成功: %1, 推理设备=%2").arg(modelPath, QString::fromStdString(recognizer_->deviceName())));
    } catch (const std::exception& error) {
        recognizer_ = nullptr;
        setRecognitionEnabled(false);
        resultLabel_->setText(QString("识别结果：模型加载失败 - %1").arg(error.what()));
        appendLog(QString("模型加载失败: %1").arg(error.what()));
    }

    setRecognitionBusy(false);
}

void MainWindow::onModelSelectionChanged(int)
{
    loadRecognizerForSelection();
}

void MainWindow::onAirModeToggled(bool checked)
{
    airWritingEnabled_ = checked;
    airSmoothedPointValid_ = false;
    if (airModeButton_ != nullptr) {
        airModeButton_->setText(checked ? "关闭隔空书写" : "开启隔空书写");
    }

    if (checked) {
        if (startCameraPreview() && startAirWriting()) {
            appendLog("隔空书写已开启");
        } else {
            airWritingEnabled_ = false;
            if (airModeButton_ != nullptr) {
                airModeButton_->blockSignals(true);
                airModeButton_->setChecked(false);
                airModeButton_->blockSignals(false);
                airModeButton_->setText("开启隔空书写");
            }
            stopAirWriting();
            stopCameraPreview();
            appendLog("隔空书写启动失败");
        }
    } else {
        stopAirWriting();
        stopCameraPreview();
        appendLog("隔空书写已关闭");
    }
}

void MainWindow::onCameraSelectionChanged(int index)
{
    if (airController_ == nullptr || index < 0) {
        return;
    }

    const int cameraIndex = cameraComboBox_ != nullptr ? cameraComboBox_->itemData(index).toInt() : index;
    const QString rawName = cameraComboBox_ != nullptr
                                ? cameraComboBox_->itemData(index, Qt::UserRole + 1).toString().trimmed()
                                : QString();
    const QString selectedName = (!rawName.isEmpty() && cameraComboBox_ != nullptr)
                                     ? rawName
                                     : (cameraComboBox_ != nullptr ? cameraComboBox_->itemText(index) : QString());
    airController_->setCameraSelection(cameraIndex, selectedName);
    if (airController_->isRunning()) {
        stopAirWriting();
        stopCameraPreview();
        if (airModeButton_ != nullptr) {
            airModeButton_->blockSignals(true);
            airModeButton_->setChecked(false);
            airModeButton_->blockSignals(false);
            airModeButton_->setText("开启隔空书写");
        }
        airWritingEnabled_ = false;
        const QString selectedName = cameraComboBox_ != nullptr ? cameraComboBox_->itemText(index) : QString::number(cameraIndex);
        appendLog(QString("摄像头已切换到 %1，需要重新开启隔空书写").arg(selectedName));
    }
}

void MainWindow::onPreviewMirrorToggled(bool checked)
{
    mirrorCameraPreview_ = checked;
    appendLog(checked ? QStringLiteral("摄像头画面: 镜像") : QStringLiteral("摄像头画面: 原始"));
}

bool MainWindow::startCameraPreview()
{
    stopCameraPreview();

    const QList<QCameraDevice> cameras = QMediaDevices::videoInputs();
    if (cameras.isEmpty()) {
        appendLog(QStringLiteral("未检测到可用摄像头"));
        return false;
    }

    const int selectedIndex = cameraComboBox_ != nullptr ? cameraComboBox_->currentData().toInt() : 0;
    const int cameraIndex = (selectedIndex >= 0 && selectedIndex < cameras.size()) ? selectedIndex : 0;

    captureSession_ = new QMediaCaptureSession();
    videoSink_ = new QVideoSink(this);
    camera_ = new QCamera(cameras.at(cameraIndex), this);

    connect(videoSink_, &QVideoSink::videoFrameChanged, this, &MainWindow::onCameraFrameChanged);

    captureSession_->setCamera(camera_);
    captureSession_->setVideoSink(videoSink_);

    camera_->start();
    appendLog(QStringLiteral("Qt 原生摄像头已启动: %1").arg(cameras.at(cameraIndex).description()));
    return true;
}

void MainWindow::stopCameraPreview()
{
    if (camera_ != nullptr) {
        camera_->stop();
        camera_->deleteLater();
        camera_ = nullptr;
    }

    if (videoSink_ != nullptr) {
        videoSink_->disconnect(this);
        videoSink_->deleteLater();
        videoSink_ = nullptr;
    }

    if (captureSession_ != nullptr) {
        captureSession_->setVideoSink(nullptr);
        captureSession_->setCamera(nullptr);
        delete captureSession_;
        captureSession_ = nullptr;
    }

    if (cameraPreviewLabel_ != nullptr) {
        cameraPreviewLabel_->setPixmap(QPixmap());
        cameraPreviewLabel_->setText("摄像头未开启");
    }
}

void MainWindow::onCameraFrameChanged(const QVideoFrame& frame)
{
    if (!frame.isValid()) {
        return;
    }

    QImage image = frame.toImage();
    if (image.isNull()) {
        return;
    }

    if (mirrorCameraPreview_) {
        image = image.flipped(Qt::Horizontal);
    }

    if (airWritingEnabled_ && airController_ != nullptr) {
        airController_->submitFrame(image);
    }

    renderCameraPreview(image);
}

void MainWindow::renderCameraPreview(const QImage& image)
{
    if (cameraPreviewLabel_ == nullptr || image.isNull()) {
        return;
    }

    QImage preview = image;
    if (trackerCursorValid_ && !trackerFrameSize_.isEmpty()) {
        const double xRatio = static_cast<double>(preview.width()) / static_cast<double>(trackerFrameSize_.width());
        const double yRatio = static_cast<double>(preview.height()) / static_cast<double>(trackerFrameSize_.height());
        const QPointF previewPoint(trackerCursorPoint_.x() * xRatio, trackerCursorPoint_.y() * yRatio);

        QPainter painter(&preview);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(QPen(Qt::white, 3));
        painter.setBrush(trackerDrawingActive_ ? QColor(0, 220, 120, 180) : QColor(255, 140, 0, 180));
        painter.drawEllipse(previewPoint, 14.0, 14.0);
        painter.setPen(QPen(Qt::black, 1));
        painter.drawText(18, 28, trackerDrawingActive_ ? QStringLiteral("DRAW") : QStringLiteral("TRACK"));
    }

    cameraPreviewLabel_->setPixmap(QPixmap::fromImage(preview).scaled(
        cameraPreviewLabel_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
}

void MainWindow::onAirTrackingUpdated(const QPointF& cursorPoint, const QSize& frameSize, bool drawingActive)
{
    if (!airWritingEnabled_ || canvas_ == nullptr || frameSize.isEmpty()) {
        return;
    }

    trackerCursorPoint_ = cursorPoint;
    trackerFrameSize_ = frameSize;
    trackerCursorValid_ = true;
    trackerDrawingActive_ = drawingActive;

    const QPointF canvasPointF = mapCameraToCanvas(cursorPoint, frameSize, canvas_->size());
    if (!airSmoothedPointValid_) {
        airSmoothedPoint_ = canvasPointF;
        airSmoothedPointValid_ = true;
    } else {
        const double alpha = 0.35;
        airSmoothedPoint_ = QPointF(
            airSmoothedPoint_.x() * (1.0 - alpha) + canvasPointF.x() * alpha,
            airSmoothedPoint_.y() * (1.0 - alpha) + canvasPointF.y() * alpha);
    }

    const QPointF drawPoint = airSmoothedPoint_;
    const QPoint drawPointInt = drawPoint.toPoint();

    if (drawingActive) {
        if (!airStrokeActive_) {
            canvas_->beginStroke(drawPointInt);
            airStrokeActive_ = true;
            lastAirPoint_ = drawPoint;
            return;
        }

        const QPointF drawDelta = drawPoint - lastAirPoint_;
        const double drawDistance = std::hypot(drawDelta.x(), drawDelta.y());
        if (drawDistance > 150.0) {
            finalizeAirStroke();
            airSmoothedPointValid_ = true;
            lastAirPoint_ = drawPoint;
            return;
        }

        if (drawDistance > 6.0 && std::isfinite(drawDistance) && drawDistance > 0.0) {
            const int steps = static_cast<int>(std::min(12.0, std::max(2.0, drawDistance / 6.0)));
            for (int s = 1; s <= steps; ++s) {
                const QPoint interp = QPoint(
                    static_cast<int>(lastAirPoint_.x() + drawDelta.x() * (static_cast<double>(s) / steps)),
                    static_cast<int>(lastAirPoint_.y() + drawDelta.y() * (static_cast<double>(s) / steps)));
                canvas_->appendStroke(interp);
            }
        } else {
            canvas_->appendStroke(drawPointInt);
        }

        lastAirPoint_ = drawPoint;
    } else if (airStrokeActive_) {
        finalizeAirStroke();
        lastAirPoint_ = drawPoint;
    } else {
        lastAirPoint_ = drawPoint;
    }
}

void MainWindow::onAirTrackingLost()
{
    airSmoothedPointValid_ = false;
    trackerCursorValid_ = false;
    trackerDrawingActive_ = false;
    finalizeAirStroke();
}

bool MainWindow::startAirWriting()
{
    if (airController_ != nullptr) {
        airController_->start();
        return airController_->isRunning();
    }

    return false;
}

void MainWindow::stopAirWriting()
{
    finalizeAirStroke();
    airSmoothedPointValid_ = false;
    if (airController_ != nullptr) {
        airController_->stop();
    }
}

QPointF MainWindow::mapCameraToCanvas(const QPointF& cameraPos, const QSize& cameraFrameSize, const QSize& canvasSize) const
{
    if (cameraFrameSize.isEmpty() || canvasSize.isEmpty()) {
        return QPointF();
    }

    const double xRatio = static_cast<double>(canvasSize.width()) / static_cast<double>(cameraFrameSize.width());
    const double yRatio = static_cast<double>(canvasSize.height()) / static_cast<double>(cameraFrameSize.height());
    const double mappedX = cameraPos.x() * xRatio;
    const double mappedY = cameraPos.y() * yRatio;

    const double clampedX = std::max(0.0, std::min(mappedX, static_cast<double>(canvasSize.width() - 1)));
    const double clampedY = std::max(0.0, std::min(mappedY, static_cast<double>(canvasSize.height() - 1)));
    return QPointF(clampedX, clampedY);
}

void MainWindow::finalizeAirStroke()
{
    if (airStrokeActive_ && canvas_ != nullptr) {
        canvas_->endStroke();
        airStrokeActive_ = false;
    }
}

void MainWindow::appendLog(const QString& message)
{
    if (logView_ == nullptr) {
        return;
    }

    const QString timestamp = QDateTime::currentDateTime().toString("HH:mm:ss");
    logView_->appendPlainText(QString("[%1] %2").arg(timestamp, message));
}

void MainWindow::onRecognize()
{
    if (recognizer_ == nullptr) {
        QMessageBox::warning(this, "提示", "模型尚未加载，无法识别。");
        appendLog("点击识别: 模型未加载");
        return;
    }

    if (recognitionBusy_) {
        appendLog("点击识别: 正在识别中，已忽略重复点击");
        return;
    }

    const QImage image = canvas_->getImage().toImage().convertToFormat(QImage::Format_Grayscale8);
    if (image.isNull() || !hasVisibleInk(image)) {
        QMessageBox::information(this, "提示", "请先在画板上书写数字。");
        appendLog("点击识别: 画板为空");
        return;
    }

    setRecognitionBusy(true);
    struct BusyReset {
        MainWindow* window;
        ~BusyReset() { window->setRecognitionBusy(false); }
    } busyReset{this};

    try {
        appendLog(QString("开始识别, 图像尺寸=%1x%2").arg(image.width()).arg(image.height()));
        const int digit = recognizer_->predict(image);
        resultLabel_->setText(QString("识别结果：%1").arg(digit));
        appendLog(QString("识别完成, 结果=%1").arg(digit));
    } catch (const std::exception& error) {
        QMessageBox::critical(this, "识别失败", error.what());
        appendLog(QString("识别异常: %1").arg(error.what()));
    } catch (...) {
        QMessageBox::critical(this, "识别失败", "发生未知异常，请查看日志。");
        appendLog("识别异常: unknown exception");
    }
}

void MainWindow::onClear()
{
    canvas_->clear();
    resultLabel_->setText("识别结果：");
    appendLog("清空画板");
}

void MainWindow::refreshCameraList()
{
    if (cameraComboBox_ == nullptr || airController_ == nullptr) {
        return;
    }

    cameraComboBox_->blockSignals(true);
    cameraComboBox_->clear();

    const QList<QCameraDevice> cameras = QMediaDevices::videoInputs();
    if (!cameras.isEmpty()) {
        for (int i = 0; i < cameras.size(); ++i) {
            const QString description = cameras.at(i).description().trimmed();
            const QString displayName = description.isEmpty() ? QStringLiteral("Camera %1").arg(i + 1) : description;
            cameraComboBox_->addItem(displayName, i);
        }
        cameraComboBox_->setEnabled(true);
        const int selectedCamera = airController_->cameraIndex();
        const int selectedUiIndex = std::clamp(selectedCamera, 0, cameraComboBox_->count() - 1);
        cameraComboBox_->setCurrentIndex(selectedUiIndex);
        airController_->setCameraSelection(cameraComboBox_->itemData(selectedUiIndex).toInt(), cameraComboBox_->itemText(selectedUiIndex));
        appendLog(QString("检测到 %1 个摄像头").arg(cameras.size()));
    } else {
        cameraComboBox_->addItem("未检测到摄像头");
        cameraComboBox_->setEnabled(false);
        appendLog(QStringLiteral("未检测到可用摄像头"));
    }

    cameraComboBox_->blockSignals(false);
}
