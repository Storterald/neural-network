#pragma once

#include <string_view>
#include <filesystem>
#include <exception>
#include <fstream>
#include <string>

namespace nn {

enum log_type {
        LOG_INFO,
        LOG_DEBUG,
        LOG_ERROR,
        LOG_FATAL

}; // enum log_type

class logger {
public:
        static constexpr uint32_t MAX_FILE_SIZE = (uint32_t)1e9;

        static logger &log()
        {
                static logger log{};
                return log;
        }

        void set_directory(const std::filesystem::path &path);
        void set_print_on_fatal(bool value);

        static std::string pref(log_type type, std::string_view file, int line);

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

        ~logger();

private:
        std::filesystem::path        m_dir;
        std::ofstream                m_file;
        uint32_t                     m_fileCount;
        bool                         m_printOnFatal;

        logger();

        void _update_file();

}; // class logger

} // namespace nn

#define LOGGER_PREF(type) ::nn::logger::pref(::nn::LOG_##type, __FILE__, __LINE__)
#define LOGGER_EX(msg) ::nn::logger::fatal_error(LOGGER_PREF(FATAL) + msg)
