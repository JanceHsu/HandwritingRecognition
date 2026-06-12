#pragma once

#include <QElapsedTimer>
#include <QMainWindow>
#include <QPointF>
#include <QSize>
#include <QString>

class Canvas;
class AirWriteController;
class DigitRecognizer;
class QLabel;
class QPlainTextEdit;
class QPushButton;
class QComboBox;
class QCheckBox;
class QImage;
class QVideoFrame;
class QCamera;
class QVideoSink;
class QMediaCaptureSession;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onRecognize();
    void onClear();
    void onCudaToggled(bool checked);
    void onAirModeToggled(bool checked);
    void onCameraSelectionChanged(int index);
    void onPreviewMirrorToggled(bool checked);
    void onCameraFrameChanged(const QVideoFrame& frame);
    void onAirTrackingUpdated(const QPointF& cursorPoint, const QSize& frameSize, bool drawingActive);
    void onAirTrackingMetricsUpdated(float confidence, int gesture, bool indexTrusted);
    void onAirTrackingLost();

private:
    void buildUi();
    void bindEvents();
    void setRecognitionEnabled(bool enabled);
    void appendLog(const QString& message);
    void setRecognitionBusy(bool busy);
    QString resolveModelPath() const;
    void loadRecognizer();
    bool startAirWriting();
    void stopAirWriting();
    bool startCameraPreview();
    void stopCameraPreview();
    QPointF mapCameraToCanvas(const QPointF& cameraPos, const QSize& cameraFrameSize, const QSize& canvasSize) const;
    void finalizeAirStroke();
    void refreshCameraList();
    void renderCameraPreview(const QImage& image);

    Canvas* canvas_ = nullptr;
    QLabel* cameraPreviewLabel_ = nullptr;
    QLabel* trackingConfidenceLabel_ = nullptr;
    QLabel* trackingGestureLabel_ = nullptr;
    QLabel* trackingLockLabel_ = nullptr;
    QLabel* cameraLabel_ = nullptr;
    QLabel* resultLabel_ = nullptr;
    QLabel* confidenceLabel_ = nullptr;
    QComboBox* cameraComboBox_ = nullptr;
    QCheckBox* cudaCheckBox_ = nullptr;
    QCheckBox* mirrorPreviewCheckBox_ = nullptr;
    QPushButton* airModeButton_ = nullptr;
    QPushButton* recognizeButton_ = nullptr;
    QPushButton* clearButton_ = nullptr;
    QPlainTextEdit* logView_ = nullptr;
    AirWriteController* airController_ = nullptr;
    DigitRecognizer* recognizer_ = nullptr;
    QMediaCaptureSession* captureSession_ = nullptr;
    QCamera* camera_ = nullptr;
    QVideoSink* videoSink_ = nullptr;
    bool airWritingEnabled_ = false;
    bool airStrokeActive_ = false;
    QPointF lastAirPoint_;
    bool mirrorCameraPreview_ = false;
    bool cudaAvailable_ = false;
    bool airSmoothedPointValid_ = false;
    QPointF airSmoothedPoint_;
    QPointF trackerCursorPoint_;
    QSize trackerFrameSize_;
    bool trackerCursorValid_ = false;
    bool trackerDrawingActive_ = false;
    QElapsedTimer airStrokeReleaseTimer_;
    bool airStrokeReleasePending_ = false;
    bool recognitionBusy_ = false;
};
