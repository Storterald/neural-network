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
extern class Logger {
private:
        uint32_t fileCount { 0 };
        std::ofstream file;

        static std::string _time();

public:
        Logger();
        ~Logger();

        // Gives the log string prefix, contains time, log type, thread and application.
        template<LogType type = DEBUG>
        static inline std::string pref()
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

        struct fatal_error : std::exception {
                explicit fatal_error(const char* message);
        };

        template<typename T>
        inline std::ofstream &operator<< (const T &value)
        {
                // Limits the log file size to avoid IDEs crashes.
                if (file.tellp() >= MAX_FILE_SIZE) {
                        file.close();
                        file = std::ofstream(BASE_PATH "/logs/latest-" + std::to_string(++fileCount) + ".log");
                }

                file << value;
                return file;
        }

} Log;