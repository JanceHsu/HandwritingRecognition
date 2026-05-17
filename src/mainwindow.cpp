#include "mainwindow.h"

#include "canvas.h"
#include "recognizer.h"

#include <QApplication>
#include <QCoreApplication>
#include <QFrame>
#include <QHBoxLayout>
#include <QDateTime>
#include <QImage>
#include <QDir>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QPushButton>
#include <QFileInfo>
#include <QComboBox>
#include <QStatusBar>
#include <QVBoxLayout>

#include <torch/cuda.h>

#include <memory>

namespace {

double averageIntensity(const QImage& img)
{
    if (img.isNull()) {
        return 255.0;
    }

    double sum = 0.0;
    const int width = img.width();
    const int height = img.height();
    for (int row = 0; row < height; ++row) {
        const auto* line = img.constScanLine(row);
        for (int col = 0; col < width; ++col) {
            sum += static_cast<unsigned char>(line[col]);
        }
    }

    return sum / static_cast<double>(width * height);
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    buildUi();
    bindEvents();
    appendLog("程序启动");

    appendLog(QString("CUDA可用: %1").arg(torch::cuda::is_available() ? "true" : "false"));
    appendLog(QString("推理设备环境变量 LIBTORCH_DEVICE=%1").arg(qEnvironmentVariable("LIBTORCH_DEVICE", "<unset>")));

    loadRecognizerForSelection();
}

MainWindow::~MainWindow()
{
    delete recognizer_;
}

void MainWindow::buildUi()
{
    setWindowTitle("手写数字识别");
    resize(760, 420);

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

    auto* canvasCard = new QFrame(centralWidget);
    canvasCard->setObjectName("card");
    auto* canvasLayout = new QVBoxLayout(canvasCard);
    canvasLayout->setContentsMargins(18, 18, 18, 18);

    auto* titleLabel = new QLabel("手写画板", canvasCard);
    titleLabel->setObjectName("title");
    canvas_ = new Canvas(canvasCard);
    canvas_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    canvasLayout->addWidget(titleLabel);
    canvasLayout->addWidget(canvas_, 1);

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

    auto* hintLabel = new QLabel("在左侧画板中写下 0 到 9 的数字，然后点击识别。", sideCard);
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
    sideLayout->addWidget(modelLabel_);
    sideLayout->addWidget(modelComboBox_);
    sideLayout->addWidget(hintLabel);
    sideLayout->addStretch(1);
    sideLayout->addWidget(recognizeButton_);
    sideLayout->addWidget(clearButton_);
    sideLayout->addWidget(logLabel);
    sideLayout->addWidget(logView_, 1);

    rootLayout->addWidget(canvasCard, 2);
    rootLayout->addWidget(sideCard, 1);

    setCentralWidget(centralWidget);

    // Footer / attribution (ensure visible and aligned right)
    auto* footer = new QLabel("Made by Jance", this);
    footer->setStyleSheet("color: #6b6b6b; font-size: 11px;");
    footer->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    footer->setFixedHeight(18);
    statusBar()->addPermanentWidget(footer, 1);
    statusBar()->setSizeGripEnabled(false);
    statusBar()->show();
}

void MainWindow::bindEvents()
{
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
    const QDir appDir(QCoreApplication::applicationDirPath());
    const QString modelRoot = appDir.filePath("models");
    const QString gpuPath = QDir(modelRoot).filePath("gpu/mnist_model.pt");
    const QString cpuPath = QDir(modelRoot).filePath("cpu/mnist_model.pt");

    if (modelKey == "gpu") {
        if (QFileInfo::exists(gpuPath)) {
            return gpuPath;
        }
        return cpuPath;
    }

    if (QFileInfo::exists(cpuPath)) {
        return cpuPath;
    }
    return gpuPath;
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
    if (image.isNull() || averageIntensity(image) > 250.0) {
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