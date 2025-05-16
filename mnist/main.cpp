#include <filesystem>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>

namespace fs = std::filesystem;

static void _parse(fs::path path)
{
        std::ifstream file(path.replace_extension(".csv"));
        if (!file)
                throw std::runtime_error("Input file " + path.string() + ".csv does not exist.");

        std::ofstream binaryFile(path.replace_extension(".nntv"), std::ios::binary);

        std::string line;
        while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string token;
                std::vector<int> data;

                // Read the label
                std::getline(ss, token, ',');
                int label = std::stoi(token);

                // Read the pixels
                while (std::getline(ss, token, ','))
                        data.push_back(std::stoi(token));

                std::vector<float> fixedData(data.size());
                for (uint32_t i = 0; i < 784; ++i)
                        fixedData[i] = (float)data[i] / 255.0f;

                binaryFile.write((char *)&label, sizeof(int));
                binaryFile.write((char *)fixedData.data(), 784 * sizeof(float));
        }

        binaryFile.close();
        file.close();

        std::cout << "File " << path.stem().replace_extension(".csv") << " data written to " << path.stem().replace_extension(".nntv") << std::endl;
}

int main() {
        const fs::path dir = fs::path(__FILE__).parent_path();
        _parse(dir / "mnist_train");
        _parse(dir / "mnist_test");
        return 0;
}
