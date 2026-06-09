#include "mainwindow.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>

namespace {

void primeRuntimeSearchPaths()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    QStringList pathParts = QString::fromLocal8Bit(qgetenv("PATH")).split(';', Qt::SkipEmptyParts);
    const QString nativeAppDir = QDir::toNativeSeparators(appDir);
    if (!pathParts.contains(nativeAppDir, Qt::CaseInsensitive)) {
        pathParts.prepend(nativeAppDir);
    }

    qputenv("PATH", pathParts.join(';').toLocal8Bit());
    QCoreApplication::addLibraryPath(appDir);
}

} // namespace

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    primeRuntimeSearchPaths();
    QApplication::setStyle("Fusion");
    QApplication::setApplicationName("Handwriting Recognition");
    QApplication::setApplicationDisplayName("Handwriting Recognition");

    MainWindow window;
    window.show();

    return app.exec();
}