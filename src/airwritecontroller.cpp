#include "airwritecontroller.h"

#include <QBuffer>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QProcessEnvironment>
#include <QtEndian>

#include <algorithm>
#include <cstring>

namespace {

QString searchScriptPath(const QStringList& candidates)
{
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return {};
}

QPointF pointFromArray(const QJsonValue& value)
{
    const QJsonArray array = value.toArray();
    if (array.size() < 2) {
        return {};
    }
    return QPointF(array.at(0).toDouble(), array.at(1).toDouble());
}

QSize sizeFromArray(const QJsonValue& value)
{
    const QJsonArray array = value.toArray();
    if (array.size() < 2) {
        return {};
    }
    return QSize(array.at(0).toInt(), array.at(1).toInt());
}

} // namespace

QString AirWriteController::pythonExecutable()
{
    const QString envPython = qEnvironmentVariable("HANDWRITING_RECOG_PYTHON");
    if (!envPython.isEmpty() && QFileInfo::exists(envPython)) {
        return QFileInfo(envPython).absoluteFilePath();
    }

    const QString homePython = QDir::home().filePath("AppData/Local/Programs/Python/Python313/python.exe");
    if (QFileInfo::exists(homePython)) {
        return QFileInfo(homePython).absoluteFilePath();
    }

    return QStringLiteral("python");
}

QString AirWriteController::trackerScriptPath()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).filePath("scripts/hand_tracker_service.py"),
        QDir(appDir).filePath("../scripts/hand_tracker_service.py"),
        QDir(appDir).filePath("../../scripts/hand_tracker_service.py"),
        QDir(appDir).filePath("../../../scripts/hand_tracker_service.py"),
    };
    return searchScriptPath(candidates);
}

AirWriteController::AirWriteController(QObject* parent)
    : QObject(parent)
{
    trackerProcess_.setProcessChannelMode(QProcess::SeparateChannels);

    connect(&trackerProcess_, &QProcess::readyReadStandardOutput,
            this, &AirWriteController::handleReadyReadStdOut);
    connect(&trackerProcess_, &QProcess::readyReadStandardError,
            this, &AirWriteController::handleReadyReadStdErr);
    connect(&trackerProcess_, &QProcess::errorOccurred,
            this, &AirWriteController::handleProcessError);
    connect(&trackerProcess_,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &AirWriteController::handleProcessFinished);
}

AirWriteController::~AirWriteController()
{
    stop();
}

bool AirWriteController::isRunning() const
{
    return running_.load();
}

int AirWriteController::cameraIndex() const
{
    return selectedCameraIndex_.load();
}

void AirWriteController::setCameraIndex(int index)
{
    selectedCameraIndex_.store(std::max(0, index));
    selectedCameraName_.clear();
}

void AirWriteController::setCameraSelection(int index, const QString& name)
{
    selectedCameraIndex_.store(std::max(0, index));
    selectedCameraName_ = name.trimmed();
}

void AirWriteController::clearProcessState()
{
    stdoutBuffer_.clear();
    trackingActive_ = false;
    trackingEmitTimer_.invalidate();
    frameEmitTimer_.invalidate();
    pendingFrame_ = QImage();
}

void AirWriteController::submitFrame(const QImage& image)
{
    if (!running_.load() || image.isNull()) {
        return;
    }

    pendingFrame_ = image;
    trySendPendingFrame();
}

void AirWriteController::trySendPendingFrame()
{
    if (!running_.load() || trackerProcess_.state() != QProcess::Running || pendingFrame_.isNull()) {
        return;
    }

    if (frameEmitTimer_.isValid() && frameEmitTimer_.elapsed() < frameSubmitIntervalMs_) {
        return;
    }

    if (trackerProcess_.bytesToWrite() > (512 * 1024)) {
        return;
    }

    QImage frame = pendingFrame_;
    pendingFrame_ = QImage();

    if (frameMaxWidth_ > 0 && frame.width() > frameMaxWidth_) {
        const double scale = static_cast<double>(frameMaxWidth_) / static_cast<double>(frame.width());
        frame = frame.scaled(frameMaxWidth_, std::max(1, static_cast<int>(std::round(frame.height() * scale))),
                             Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }

    QByteArray jpegBytes;
    QBuffer buffer(&jpegBytes);
    if (!buffer.open(QIODevice::WriteOnly)) {
        return;
    }

    if (!frame.save(&buffer, "JPG", frameJpegQuality_)) {
        return;
    }

    QByteArray packet;
    packet.reserve(4 + jpegBytes.size());
    char header[4];
    qToBigEndian<quint32>(static_cast<quint32>(jpegBytes.size()), reinterpret_cast<uchar*>(header));
    packet.append(header, 4);
    packet.append(jpegBytes);

    const qint64 written = trackerProcess_.write(packet);
    if (written <= 0) {
        pendingFrame_ = frame;
        emit statusMessage(QStringLiteral("向 Python 发送摄像头帧失败"));
        return;
    }

    frameEmitTimer_.start();
}

void AirWriteController::start()
{
    if (running_.exchange(true)) {
        return;
    }

    if (trackerProcess_.state() != QProcess::NotRunning) {
        trackerProcess_.kill();
        trackerProcess_.waitForFinished(1000);
    }

    clearProcessState();

    const QString python = pythonExecutable();
    const QString script = trackerScriptPath();
    if (script.isEmpty()) {
        running_.store(false);
        emit statusMessage(QStringLiteral("未找到 Python 隔空追踪脚本"));
        return;
    }

    const int selectedSlot = selectedCameraIndex_.load();
    const QString selectedName = selectedCameraName_.trimmed();
    if (!selectedName.isEmpty()) {
        emit statusMessage(QStringLiteral("摄像头选择: %1").arg(selectedName));
    } else {
        emit statusMessage(QStringLiteral("摄像头索引: %1").arg(selectedSlot));
    }

    QStringList arguments = {
        script,
        QStringLiteral("--stdin-frames"),
        QStringLiteral("--tracking-fps"), QStringLiteral("45"),
        QStringLiteral("--detect-max-width"), QStringLiteral("512"),
    };

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("TF_CPP_MIN_LOG_LEVEL", "3");
    env.insert("GLOG_minloglevel", "2");
    trackerProcess_.setProcessEnvironment(env);

    trackerProcess_.start(python, arguments);
    if (!trackerProcess_.waitForStarted(4000)) {
        running_.store(false);
        emit statusMessage(QStringLiteral("无法启动 Python 隔空追踪进程"));
        return;
    }

    emit statusMessage(QStringLiteral("Python 隔空追踪服务已启动"));
}

void AirWriteController::stop()
{
    const bool wasRunning = running_.exchange(false);

    pendingFrame_ = QImage();

    if (trackerProcess_.state() != QProcess::NotRunning) {
        trackerProcess_.closeWriteChannel();
        trackerProcess_.terminate();
        if (!trackerProcess_.waitForFinished(1500)) {
            trackerProcess_.kill();
            trackerProcess_.waitForFinished(1000);
        }
    }

    clearProcessState();

    if (wasRunning) {
        emit trackingLost();
    }
}

void AirWriteController::emitParsedLine(const QByteArray& line)
{
    const QByteArray trimmed = line.trimmed();
    if (trimmed.isEmpty()) {
        return;
    }

    const QJsonDocument document = QJsonDocument::fromJson(trimmed);
    if (!document.isObject()) {
        return;
    }

    const QJsonObject object = document.object();
    const QString type = object.value(QStringLiteral("type")).toString();

    if (type == QLatin1String("status")) {
        const QString message = object.value(QStringLiteral("message")).toString();
        if (!message.isEmpty()) {
            emit statusMessage(message);
        }
        return;
    }

    if (type != QLatin1String("frame")) {
        return;
    }

    const bool hasHand = object.value(QStringLiteral("has_hand")).toBool(false);
    const bool drawingActive = object.value(QStringLiteral("drawing_active")).toBool(false);
    const QSize frameSize = sizeFromArray(object.value(QStringLiteral("frame_size")));
    if (!hasHand) {
        if (trackingActive_) {
            trackingActive_ = false;
            emit trackingLost();
        }
        return;
    }

    const QPointF cursorPoint = pointFromArray(object.value(QStringLiteral("cursor")));
    trackingActive_ = true;
    if (!trackingEmitTimer_.isValid() || trackingEmitTimer_.elapsed() >= trackingEmitIntervalMs_) {
        trackingEmitTimer_.start();
        emit trackingUpdated(cursorPoint, frameSize, drawingActive);
    }
}

void AirWriteController::handleReadyReadStdOut()
{
    stdoutBuffer_.append(trackerProcess_.readAllStandardOutput());

    int newlineIndex = -1;
    while ((newlineIndex = stdoutBuffer_.indexOf('\n')) >= 0) {
        const QByteArray line = stdoutBuffer_.left(newlineIndex);
        stdoutBuffer_.remove(0, newlineIndex + 1);
        emitParsedLine(line);
    }
}

void AirWriteController::handleReadyReadStdErr()
{
    // MediaPipe/TensorFlow 的 C++ 后端会向 stderr 输出 INFO/WARNING 级别的初始化日志，
    // 这些不是真正的错误，通过环境变量 TF_CPP_MIN_LOG_LEVEL 和 GLOG_minloglevel 抑制。
    // 此处不再将 stderr 逐行转发到 UI 日志。
    trackerProcess_.readAllStandardError();
}

void AirWriteController::handleProcessError(QProcess::ProcessError error)
{
    if (error == QProcess::FailedToStart) {
        running_.store(false);
        emit statusMessage(QStringLiteral("Python 隔空追踪进程启动失败"));
        emit trackingLost();
    }
}

void AirWriteController::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    Q_UNUSED(exitCode);
    Q_UNUSED(exitStatus);

    const bool wasRunning = running_.exchange(false);
    if (wasRunning) {
        emit trackingLost();
    }
}
