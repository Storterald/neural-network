#pragma once

#include <string_view>
#include <filesystem>
#include <exception>
#include <fstream>
#include <string>

#include <neural-network/Base.h>

NN_BEGIN

enum LogType {
        LOG_INFO,
        LOG_DEBUG,
        LOG_ERROR,
        LOG_FATAL

}; // enum LogType

class Logger {
public:
        static constexpr uint32_t MAX_FILE_SIZE = (uint32_t)1e9;

        static Logger &Log()
        {
                static Logger log{};
                return log;
        }

        void set_directory(const std::filesystem::path &path);

        static std::string pref(LogType type, std::string_view file, int line);

        struct fatal_error final : std::exception {
                explicit fatal_error(const std::string &message);
        };

        template<typename T>
        std::ofstream &operator<< (const T &value)
        {
                _update_file();
                m_file << value;
                return m_file;
        }

        ~Logger();

private:
        std::filesystem::path        m_dir;
        uint32_t                     m_fileCount;
        std::ofstream                m_file;

        Logger();

        void _update_file();

}; // class Logger

NN_END

#define LOGGER_PREF(type) NN Logger::pref(NN LOG_##type, __FILE__, __LINE__)
#define LOGGER_EX(msg) NN Logger::fatal_error(LOGGER_PREF(FATAL) + msg)
