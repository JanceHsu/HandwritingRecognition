#pragma once

#include <QMainWindow>
#include <QString>

class Canvas;
class DigitRecognizer;
class QLabel;
class QPlainTextEdit;
class QPushButton;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onRecognize();
    void onClear();
    void onModelSelectionChanged(int index);

private:
    void buildUi();
    void bindEvents();
    void setRecognitionEnabled(bool enabled);
    void appendLog(const QString& message);
    void setRecognitionBusy(bool busy);
    QString selectedModelKey() const;
    QString resolveModelPath(const QString& modelKey) const;
    void loadRecognizerForSelection();

    Canvas* canvas_ = nullptr;
    QLabel* resultLabel_ = nullptr;
    QLabel* modelLabel_ = nullptr;
    class QComboBox* modelComboBox_ = nullptr;
    QPushButton* recognizeButton_ = nullptr;
    QPushButton* clearButton_ = nullptr;
    QPlainTextEdit* logView_ = nullptr;
    DigitRecognizer* recognizer_ = nullptr;
    bool recognitionBusy_ = false;
};