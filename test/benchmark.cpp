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
//BENCHMARK(benchMark_hourGlass);

void benchMark_hourGlass_calculatePhysics(benchmark::State& state) {
	for (auto _ : state) {
		Scene scene;
		// TODO: create and connect a MainWindow, or eliminate it (it is a nullptr currently)
		scene.setBenchmarkMode(true);

		scene.applyDefaults();
		int N = scene.getDefaults().number_of_particles;
		scene.setNumberOfParticles(N);
		scene.addParticles(scene.getNumberOfParticles());

		scene.createCells();
		scene.populateCells();
		scene.resolveConstraintsOnInitCells(5);
		scene.populateCells();

		scene.setRunning();
		while ( scene.isRunning() ) {
			scene.calculatePhysics();
		}
	}
}

BENCHMARK(benchMark_hourGlass_calculatePhysics);

BENCHMARK_MAIN();
