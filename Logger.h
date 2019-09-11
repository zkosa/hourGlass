#ifndef LOGGER_H_
#define LOGGER_H_

#include <string>
#include <vector>

class Logger {
	std::string file_name = "log.csv";
	std::vector<std::string> header;
	std::vector<double> data;

public:
	Logger() {}

};

#endif /* LOGGER_H_ */
