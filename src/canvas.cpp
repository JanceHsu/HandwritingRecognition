#include "canvas.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>

Canvas::Canvas(QWidget* parent)
    : QWidget(parent), pixmap_(280, 280)
{
    setMinimumSize(280, 280);
    setMouseTracking(true);
    clear();
}

QPixmap Canvas::getImage() const
{
    return pixmap_;
}

void Canvas::clear()
{
    pixmap_.fill(Qt::white);
    update();
}

void Canvas::beginStroke(const QPoint& point)
{
    drawing_ = true;
    lastPoint_ = point;
    drawPointAt(point);
}

void Canvas::appendStroke(const QPoint& point)
{
    if (!drawing_) {
        beginStroke(point);
        return;
    }

    drawLineTo(point);
}

void Canvas::endStroke()
{
    drawing_ = false;
}

void Canvas::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.drawPixmap(rect(), pixmap_);
}

void Canvas::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        beginStroke(event->pos());
    }
}

void Canvas::mouseMoveEvent(QMouseEvent* event)
{
    if ((event->buttons() & Qt::LeftButton) && drawing_) {
        appendStroke(event->pos());
    }
}

void Canvas::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && drawing_) {
        appendStroke(event->pos());
        endStroke();
    }
}

void Canvas::resizeEvent(QResizeEvent* event)
{
    const QSize newSize = event->size().expandedTo(QSize(280, 280));
    ensureCanvasSize(newSize);
    QWidget::resizeEvent(event);
}

void Canvas::ensureCanvasSize(const QSize& size)
{
    if (size == pixmap_.size()) {
        return;
    }

    QPixmap resized(size);
    resized.fill(Qt::white);

    QPainter painter(&resized);
    painter.drawPixmap(0, 0, pixmap_.scaled(size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
    pixmap_ = resized;
    update();
}

void Canvas::drawLineTo(const QPoint& endPoint)
{
    QPainter painter(&pixmap_);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(Qt::black, 20, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(lastPoint_, endPoint);

    const int radius = 24;
    update(QRect(lastPoint_, endPoint).normalized().adjusted(-radius, -radius, radius, radius));
    lastPoint_ = endPoint;
}

void Canvas::drawPointAt(const QPoint& point)
{
    QPainter painter(&pixmap_);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(Qt::black, 20, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawPoint(point);
    update(QRect(point, point).adjusted(-12, -12, 12, 12));
}