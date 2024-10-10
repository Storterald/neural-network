#pragma once

#include <fstream>
#include <string>

#include "../Base.h"

enum LogType{
        INFO,
        DEBUG,
        ERROR,
        FATAL
};

// Simple logger instead of using console. Uses macro BASE_PATH
// defined in CMake as ${CMAKE_HOME_PATH}
class Logger {
public:
        Logger();
        ~Logger();

        // Gives the log string prefix, contains time, log type, thread and application.
        template<LogType type = DEBUG>
        static std::string pref()
        {
                // Converts type enum to string, constexpr switch macro
                // instead of series of constexpr if.
                std::string typeString;
                CONSTEXPR_SWITCH(type,
                        CASE(INFO,  typeString = "INFO "),
                        CASE(DEBUG, typeString = "DEBUG"),
                        CASE(ERROR, typeString = "ERROR"),
                        CASE(FATAL, typeString = "FATAL")
                );

                return _time() + " " + typeString + " [Main] [Network] ";
        }

        static std::string fixTime(double t);

        struct fatal_error final : std::exception {
                explicit fatal_error(const std::string &message);
        };

        template<typename T>
        std::ofstream &operator<< (const T &value)
        {
                // Limits the log file size to avoid IDEs crashes.
                if (m_file.tellp() >= MAX_FILE_SIZE) {
                        m_file.close();
                        m_file = std::ofstream(BASE_PATH "/logs/latest-" + std::to_string(++m_fileCount) + ".log");
                }

                m_file << value;
                return m_file;
        }

private:
        uint32_t m_fileCount { 0 };
        // Logging to a file is usually way faster and more efficient than logging to console.
        // https://stackoverflow.com/questions/6338812/printing-to-the-console-vs-writing-to-a-file-speed.
        std::ofstream m_file;

        static std::string _time();

} extern Log;