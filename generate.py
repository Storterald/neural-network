import numpy as np

# Network architecture details
input_size = 784   # 28x28 pixels flattened
hidden_layer_size1 = 16
hidden_layer_size2 = 16
output_size = 10

# Total sizes
total_weights = input_size * hidden_layer_size1 + hidden_layer_size1 * hidden_layer_size2 + hidden_layer_size2 * output_size
total_biases = hidden_layer_size1 + hidden_layer_size2 + output_size

# Xavier Initialization Limits
limit_hidden = np.sqrt(6 / (input_size + hidden_layer_size1))
limit_output = np.sqrt(6 / (hidden_layer_size2 + output_size))

# Generate weights
weights_input_hidden = np.random.uniform(-limit_hidden, limit_hidden, (hidden_layer_size1, input_size)).flatten()
weights_hidden_hidden = np.random.uniform(-limit_hidden, limit_hidden, (hidden_layer_size1, hidden_layer_size2)).flatten()
weights_hidden_output = np.random.uniform(-limit_output, limit_output, (output_size, hidden_layer_size2)).flatten()

# Concatenate all weights into a single array
all_weights = np.concatenate([weights_input_hidden, weights_hidden_hidden, weights_hidden_output])

# Generate biases, initialized to zeros
biases_hidden_1 = np.zeros(hidden_layer_size1)
biases_hidden_2 = np.zeros(hidden_layer_size2)
biases_output = np.zeros(output_size)

# Concatenate all biases into a single array
all_biases = np.concatenate([biases_hidden_1, biases_hidden_2, biases_output])

# Ensure the sizes match
assert len(all_weights) == total_weights
assert len(all_biases) == total_biases

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
        f.write('    const std::vector<float> weights {{\n'.format(total_weights))
        for i, value in enumerate(all_weights):
                f.write(f'        {value:.6f}f')
                if i < total_weights - 1:
                        f.write(',\n')
                else:
                        f.write('\n')
        f.write('    };\n\n')

        # Write all biases
        f.write('    const std::vector<float> biases {{\n'.format(total_biases))
        for i, value in enumerate(all_biases):
                f.write(f'        {value:.6f}f')
                if i < total_biases - 1:
                        f.write(',\n')
                else:
                        f.write('\n')
        f.write('    };\n')

        f.write('}\n')

print("Generated initial random values.")
