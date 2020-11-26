#include "mainwindow.h"
#include <QApplication>
#include <iostream>
#include <string>

// TODO: add command line arguments
int main(int argc, char *argv[]) {
	QApplication application(argc, argv);
	MainWindow window;

	QStringList args = application.arguments();
	bool benchmark = false;

	if (args.length() > 2) {
		std::cerr << "Wrong number of arguments. The only accepted argument is -benchmark." << std::endl;
		return 1;
	} else if (args.length() == 2) {
		if (args[1].toStdString() == std::string("-benchmark")) {
			benchmark = true;
		} else {
			std::cerr << "Wrong argument. The only accepted argument is -benchmark." << std::endl;
			return 1;
		}
	}

	window.show();

	if (benchmark) {
		window.launchBenchmark();
	}

	return application.exec();
}
