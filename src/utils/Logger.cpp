#include "Logger.h"

#include <format>
#include <ctime>
#include <cmath>

Logger Log{};

Logger::Logger()
{
        file = std::ofstream(BASE_PATH "/logs/latest-0.log");
        if (!file)
                throw std::runtime_error("Could not open log file.");
}

Logger::~Logger()
{
        file.close();
}

std::string Logger::fixTime(
        double t
) {
        const int32_t magnitude { (int32_t) std::floor(std::log10(t)) };
        if (magnitude >= 0)
                return std::format("{:.4f}", t) + "s";
        if (magnitude >= -3)
                return std::format("{:.4f}", t * 1e3) + "ms";
        if (magnitude >= -6)
                return std::format("{:.4f}", t * 1e6) + "micros";
        if (magnitude >= -9)
                return std::format("{:.4f}", t * 1e9) + "ns";

        return std::format("{:.4f}", t * 1e12) + "ps";
}

std::string Logger::_time()
{
        char buffer[80]{};
        time_t rawTime{};

        // Get time as formatted string
        time(&rawTime);
        tm tm{};
        localtime_s(&tm, &rawTime);
        std::strftime(buffer, sizeof(buffer), "%X", &tm);

        return buffer;
}

Logger::fatal_error::fatal_error(
        const char *message
) {
        // Flush forces to print everything that has not yet been printed.
        Log.file << Logger::pref<FATAL>() + message << std::flush;
}
