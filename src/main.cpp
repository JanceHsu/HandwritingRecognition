#include "mainwindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>

namespace {

void primeDllSearchPath()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    if (appDir.isEmpty()) {
        return;
    }

#ifdef Q_OS_WIN
    const QString nativeAppDir = QDir::toNativeSeparators(appDir);
    QStringList pathParts = QString::fromLocal8Bit(qgetenv("PATH")).split(';', Qt::SkipEmptyParts);
    if (!pathParts.contains(nativeAppDir, Qt::CaseInsensitive)) {
        pathParts.prepend(nativeAppDir);
    }
    qputenv("PATH", pathParts.join(';').toLocal8Bit());
#endif
}

} // namespace

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    primeDllSearchPath();

    QApplication::setStyle("Fusion");
    QApplication::setApplicationName("Handwriting Recognition");
    QApplication::setApplicationDisplayName("Handwriting Recognition");

    MainWindow window;
    window.show();

    return app.exec();
}
