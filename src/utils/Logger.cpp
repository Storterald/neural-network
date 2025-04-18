#include "Logger.h"

#include <ctime>
#include <iostream>
#include <format>

static constexpr std::string_view CONVERTER[4] {
        "INFO ",
        "DEBUG",
        "ERROR",
        "FATAL"
};

static std::string _time()
{
        char buffer[80]{};
        time_t rawTime{};

        // Get time as formatted string
        std::time(&rawTime);
        std::tm tm{};
        localtime_s(&tm, &rawTime);
        std::strftime(buffer, sizeof(buffer), "%X", &tm);

        return buffer;
}

Logger::Logger() :
        m_file(std::ofstream(BASE_PATH "/logs/latest-0.log")) {

        if (!m_file)
                throw std::runtime_error("Could not open log file.");
}

Logger::~Logger()
{
        m_file.close();
}

std::string Logger::pref(LogType type, std::string_view file, int line) {
        return std::format("{} {} {}:{} ", _time(), CONVERTER[type], file, line);
}

Logger::fatal_error::fatal_error(const std::string &message)
{

        // MAcro mainly used for disabling during tests.
#ifndef LOGGER_FATAL_ERROR_DISABLE_STDOUT
        std::cout << message << std::endl;
#endif // LOGGER_FATAL_ERROR_DISABLE_STDOUT

        // Flush (std::endl) forces to print everything that has not yet been printed.
        Logger::Log().m_file << message << std::endl;
}
