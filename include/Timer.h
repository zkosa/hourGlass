#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

class Timer {
	std::chrono::_V2::steady_clock::time_point begin;
	std::chrono::_V2::steady_clock::time_point end;
	bool running = false;

public:
	void start() {
		begin = std::chrono::steady_clock::now();
		running = true;
	}

	void stop() {
		end = std::chrono::steady_clock::now();
		running = false;
	}

	double milliSeconds() {
		if (running) {
			stop();
		}
		return std::chrono::duration_cast < std::chrono::microseconds
				> (end - begin).count() / 1000.;
	}
	double seconds() {
		return milliSeconds() / 1000.;
	}
};

#endif /* TIMER_H_ */
