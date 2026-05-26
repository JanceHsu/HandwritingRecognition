#pragma once

#include <QObject>
#include <QByteArray>
#include <QImage>
#include <QElapsedTimer>
#include <QPointF>
#include <QProcess>
#include <QSize>
#include <QString>
#include <QStringList>
#include <atomic>

class AirWriteController : public QObject {
    Q_OBJECT

public:
    explicit AirWriteController(QObject* parent = nullptr);
    ~AirWriteController() override;

    bool isRunning() const;
    int cameraIndex() const;
    void setCameraIndex(int index);
    void setCameraSelection(int index, const QString& name);

    void submitFrame(const QImage& image);

public slots:
    void start();
    void stop();

signals:
    void trackingUpdated(const QPointF& cursorPoint, const QSize& frameSize, bool drawingActive);
    void trackingLost();
    void statusMessage(const QString& message);

private:
    void handleReadyReadStdOut();
    void handleReadyReadStdErr();
    void handleProcessError(QProcess::ProcessError error);
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    static QString pythonExecutable();
    static QString trackerScriptPath();
    void trySendPendingFrame();
    void clearProcessState();
    void emitParsedLine(const QByteArray& line);

    std::atomic_int selectedCameraIndex_{0};
    QString selectedCameraName_;
    std::atomic_bool running_ = false;
    QProcess trackerProcess_;
    QByteArray stdoutBuffer_;
    QByteArray stderrBuffer_;
    bool trackingActive_ = false;
    QElapsedTimer trackingEmitTimer_;
    int trackingEmitIntervalMs_ = 50; // ms, ~20Hz default
    QImage pendingFrame_;
    QElapsedTimer frameEmitTimer_;
    int frameSubmitIntervalMs_ = 33; // ~30 FPS to Python
    int frameJpegQuality_ = 70;
    int frameMaxWidth_ = 512;
};