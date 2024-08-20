import numpy as np
import sys

if __name__ == "__main__":
        # Network architecture details
        sizes: list[int] = [int(x) for x in sys.argv[1:]]

        WEIGHTS_COUNT: int = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
        BIASES_COUNT: int = sum(sizes) - sizes[0]

        # Xavier Initialization Limits
        limit_hidden: int = np.sqrt(6 / (sizes[0] + sizes[1]))
        limit_output: int = np.sqrt(6 / (sizes[-2] + sizes[-1]))

        # Generate weights and biases
        weights = np.random.uniform(-limit_hidden, limit_hidden, WEIGHTS_COUNT).flatten()
        biases = np.zeros(BIASES_COUNT)

        # Open file for writing
        with open('decoder/Values.h', 'w') as f:
                # Write namespace and includes
                f.write('#pragma once\n'
                        '\n'
                        '#include <vector>\n'
                        '#include <cstdint>\n'
                        '\n'
                        'namespace Values {\n'
                        '\n')

                # Write all weights
                f.write('        const std::vector<float> weights {{\n                '.format(WEIGHTS_COUNT))
                for i, value in enumerate(weights):
                        f.write(f'{value:.6f}f')
                        if i < WEIGHTS_COUNT - 1:
                                f.write(', ')
                                if (i + 1) % 10 == 0:
                                        f.write('\n                ')
                        else:
                                f.write('\n')

                f.write('        };\n\n')

                # Write all biases
                f.write(f'        const std::vector<float> biases({BIASES_COUNT}, 0.0f);\n\n''}')

        print("Generated initial random values for sizes:", sizes)
