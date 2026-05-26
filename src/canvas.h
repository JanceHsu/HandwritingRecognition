#pragma once

#include <QPixmap>
#include <QWidget>

class Canvas : public QWidget {
    Q_OBJECT

public:
    explicit Canvas(QWidget* parent = nullptr);

    QPixmap getImage() const;
    void clear();
    void beginStroke(const QPoint& point);
    void appendStroke(const QPoint& point);
    void endStroke();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void ensureCanvasSize(const QSize& size);
    void drawPointAt(const QPoint& point);
    void drawLineTo(const QPoint& endPoint);

    QPixmap pixmap_;
    QPoint lastPoint_;
    bool drawing_ = false;
};