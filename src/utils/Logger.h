#pragma once

#include <fstream>
#include <string>

enum LogType{
        INFO,
        DEBUG,
        ERROR,
        FATAL
};

class Logger {
public:
        static constexpr uint32_t MAX_FILE_SIZE = 1e9;

        Logger();
        ~Logger();

        static std::string pref(LogType type, std::string_view file, int line);

        struct fatal_error final : std::exception {
                explicit fatal_error(const std::string &message);
        };

        template<typename T>
        std::ofstream &operator<< (const T &value)
        {
                // Limits the log file size to avoid IDEs crashes.
                if (((uint32_t)m_file.tellp()) >= MAX_FILE_SIZE) {
                        m_file.close();
                        m_file = std::ofstream(BASE_PATH "/logs/latest-"
                                + std::to_string(++m_fileCount) + ".log");
                }

                m_file << value;
                return m_file;
        }

private:
        uint32_t m_fileCount { 0 };
        std::ofstream m_file;

} extern Log;

#define LOGGER_PREF(type) Logger::pref(type, __FILE__, __LINE__)
#define LOGGER_EX(msg) Logger::fatal_error(LOGGER_PREF(FATAL) + msg)