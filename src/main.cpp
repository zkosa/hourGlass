#include "mainwindow.h"
#include <QApplication>

// TODO: add command line arguments
int main(int argc, char *argv[]) {
	QApplication application(argc, argv);
	MainWindow window;
	window.show();

	return application.exec();
}
