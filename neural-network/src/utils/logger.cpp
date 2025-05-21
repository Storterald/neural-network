#include <neural-network/utils/logger.h>

#include <string_view>
#include <filesystem>
#include <stdexcept>
#include <format>
#include <string>
#include <ctime>

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

namespace nn {

logger::logger() :
        m_dir(""),
        m_file(),
        m_fileCount(0),
        m_printOnFatal(false) {}

logger::~logger()
{
        m_file.close();
}

logger &logger::set_directory(const std::filesystem::path &path)
{
        m_file.close();

        m_dir = path;
        m_fileCount = 0;
        m_file = std::ofstream(m_dir / "latest-0.log");

        if (!m_file)
                throw std::runtime_error("Could not open log file.");

        return *this;
}

logger &logger::set_print_on_fatal(bool value)
{
        m_printOnFatal = value;
        return *this;
}

std::string logger::pref(log_type type)
{
        constexpr std::string_view conv[4] {
                "INFO ",
                "DEBUG",
                "ERROR",
                "FATAL"
        };

        return std::format("{} {} ", _time(), conv[type]);
}

void logger::_update_file()
{
        if (!m_file || ((uint32_t)m_file.tellp()) >= MAX_FILE_SIZE) {
                if (m_file)
                        m_file.close();

                m_file = std::ofstream(m_dir / std::format("latest-{}.log", ++m_fileCount));
                if (!m_file)
                        throw std::runtime_error("Could not open log file.");
        }
}

} // namespace nn
