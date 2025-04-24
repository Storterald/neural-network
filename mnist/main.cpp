#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>

namespace fs = std::filesystem;

int main() {
        const fs::path dir = fs::path(__FILE__).parent_path();

        std::ifstream file(dir / "mnist_test.csv");
        std::string line;

        std::ofstream binaryFile(dir / "mnist_test.nntv", std::ios::binary);

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
                for (uint32_t i { 0 }; i < 784; i++)
                        fixedData[i] = (float)data[i] / 255.0f;

                binaryFile.write((char *)&label, sizeof(int));
                binaryFile.write((char *)fixedData.data(), 784 * sizeof(float));
        }

        binaryFile.close();
        file.close();
        return 0;
}

