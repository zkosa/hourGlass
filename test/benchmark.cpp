#include <benchmark/benchmark.h>

#include "mainwindow.h"
#include <QApplication>


void benchMark_hourGlass(benchmark::State& state) {
	for (auto _ : state) {
		// argv must be composed of non const character arrays
		char arg0[] = "hourGlass_benchmark";
		char *argv[] = { &arg0[0], NULL };
		int   argc = (int)(sizeof(argv) / sizeof(argv[0])) - 1;

		QApplication application(argc, argv);
		MainWindow window;
		window.show();

		application.exec();
	}
}
// Register the function as a benchmark
BENCHMARK(benchMark_hourGlass);

BENCHMARK_MAIN();
