#pragma once

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <string>

namespace nn {

enum log_type {
        info,
        debug,
        error,
        fatal

}; // enum log_type

class logger {
public:
        static constexpr uint32_t MAX_FILE_SIZE = (uint32_t)1e9;

        static logger &log()
        {
                static logger log{};
                return log;
        }

        static std::string pref(log_type type);

        logger &set_directory(const std::filesystem::path &path);
        logger &set_print_on_fatal(bool value);

        template<typename T>
        std::ofstream &operator<< (const T &value)
        {
                static uint32_t c = 0;

                _update_file();
                m_file << value;

                if (c++ == 50) {
                        c = 0;
                        m_file.flush();
                }

                return m_file;
        }

private:
        std::filesystem::path m_dir;
        std::ofstream         m_file;
        uint32_t              m_count;
        bool                  m_print;

        logger();

        void _update_file();

        friend struct fatal_error;
}; // class logger

} // namespace nn
