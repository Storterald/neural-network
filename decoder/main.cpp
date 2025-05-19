#include <filesystem>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

enum LayerType : uint32_t {
        FULLY_CONNECTED

}; // enum LayerType

void writeVector(
        const char           *name,
        uint32_t             count,
        const float          *values,
        std::ofstream        &outFile) {

        outFile << "        std::vector<float> " << name << " /* " << count << " */ {"
                   "\n                ";

        for (uint32_t i { 0 }; i < count - 1; i++) {
                if (values[i] >= 0.0f)
                        outFile << " ";

                outFile << values[i] << ", ";

                if ((i + 1) % 10 == 0)
                        outFile << "\n                ";
        }

        if (values[count - 1] >= 0.0f)
                outFile << " ";

        outFile << values[count - 1] << "\n"
                   "        };\n"
                   "\n";
}

void decodeFullyConnectedLayer(
        const uint32_t        infos[4],
        std::ifstream         &inFile,
        std::ofstream         &outFile) {

        const uint32_t wCount = infos[2] * infos[3];
        const auto w = new float[wCount];

        inFile.read((char *)w, wCount * sizeof(float));
        writeVector("weights", wCount, w, outFile);
        delete [] w;

        const uint32_t bCount = infos[3];
        const auto b = new float[bCount];

        inFile.read((char *)b, bCount * sizeof(float));
        writeVector("biases", bCount, b, outFile);
        delete [] b;
}

int main()
{
        const fs::path dir = fs::path(__FILE__).parent_path();

        std::ifstream inFile(dir / ".." / "mock" / "Encoded.nnv", std::ios::binary);
        if (!inFile)
                return EXIT_FAILURE;

        std::ofstream outFile(dir / "Decoded.h");
        if (!outFile)
                return EXIT_FAILURE;

        // Decoded file gotta be pretty
        outFile << std::fixed << std::setprecision(6);

        outFile << "#pragma once\n"
                   "\n"
                   "#include <cstdint>\n"
                   "#include <vector>\n"
                   "\n";

        uint32_t layerIndex { 0 };
        while (true) {
                uint32_t infos[4];
                inFile.read((char *)infos, 4 * sizeof(uint32_t));

                if (inFile.eof())
                        break;

                outFile << "namespace Layer" << layerIndex << " {\n"
                           "\n"
                           "        constexpr uint32_t type = " << infos[0] << ";\n"
                           "        constexpr uint32_t functionType = " << infos[1] << ";\n"
                           "\n";

                switch ((LayerType)infos[0]) {
                        case FULLY_CONNECTED:
                                decodeFullyConnectedLayer(infos, inFile, outFile);
                                break;
                        default:
                                throw std::runtime_error("Layer type not recognized.");
                }

                outFile << "}\n"
                           "\n";

                layerIndex++;
        }

        inFile.close();
        outFile.close();
        return EXIT_SUCCESS;
}
