#include <neural-network/utils/Logger.h>

#include <string_view>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <format>
#include <string>
#include <ctime>

#include <neural-network/Base.h>

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
        std::tm *tm = std::localtime(&rawTime);
        std::strftime(buffer, sizeof(buffer), "%X", tm);

        return buffer;
}

NN_BEGIN

Logger::Logger() :
        m_dir(""),
        m_fileCount(0),
        m_file() {}

Logger::~Logger()
{
        m_file.close();
}

void Logger::set_directory(const std::filesystem::path &path)
{
        m_file.close();

        m_dir = path;
        m_fileCount = 0;
        m_file = std::ofstream(m_dir / "latest-0.log");
        if (!m_file)
                throw std::runtime_error("Could not open log file.");
}

std::string Logger::pref(LogType type, std::string_view file, int line) {
        return std::format("{} {} {}:{} ", _time(), CONVERTER[type], file, line);
}

void Logger::_update_file()
{
        if (!m_file || (uint32_t)m_file.tellp() >= MAX_FILE_SIZE) {
                if (m_file)
                        m_file.close();

                m_file = std::ofstream(m_dir / std::format("latest-{}.log", ++m_fileCount));
                if (!m_file)
                        throw std::runtime_error("Could not open log file.");
        }
}

Logger::fatal_error::fatal_error(const std::string &message)
{
        std::cout << message << std::endl;
        Logger::Log().m_file << message << std::endl;
}

NN_END
