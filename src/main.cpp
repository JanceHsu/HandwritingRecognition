#include "mainwindow.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QApplication::setStyle("Fusion");
    QApplication::setApplicationName("Handwriting Recognition");
    QApplication::setApplicationDisplayName("Handwriting Recognition");

    MainWindow window;
    window.show();

    return app.exec();
}