#include <iostream>
#include <vector>
#include "../utils/utils.hpp"

using namespace std;

// Define the convolution function
void conv3d(const vector<vector<vector<vector<float>>>> &input,
            const vector<vector<vector<vector<float>>>> &kernel,
            vector<vector<vector<vector<float>>>> &output,
            int stride = 1, int padding = 1) {

    int batch_size = input.size();
    int in_channels = input[0].size();
    int out_channels = kernel.size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();
    int kernel_height = kernel[0][0].size();
    int kernel_width = kernel[0][0][0].size();

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    // Initialize the output tensor
    output.assign(batch_size, vector<vector<vector<float>>>(
                                  out_channels, vector<vector<float>>(
                                                  output_height, vector<float>(output_width, 0))));

    vector<vector<vector<vector<float>>>> padded_input(batch_size, vector<vector<vector<float>>>(
            in_channels, vector<vector<float>>(
            input_height + 2 * padding, vector<float>(input_width + 2 * padding, 0))));

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int i = 0; i < input_height; ++i) {
                for (int j = 0; j < input_width; ++j) {
                    padded_input[b][c][i + padding][j + padding] = input[b][c][i][j];
                }
            }
        }
    }
    // // Perform the convolution operation
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    float sum = 0.0;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ki = 0; ki < kernel_height; ++ki) {
                            for (int kj = 0; kj < kernel_width; ++kj) {
                                int input_i = i * stride + ki;
                                int input_j = j * stride + kj;
                                sum += padded_input[b][ic][input_i][input_j] * kernel[oc][ic][ki][kj];
                            }
                        }
                    }
                    output[b][oc][i][j] = sum;
                }
            }
        }
    }
    vector<float> output_1d;
    for (int b = 0; b < batch_size; b++)
        for (int i = 0; i < out_channels; i++)
        {
            for (int j = 0; j < output_height; j++)
            {
                for (int k = 0; k < output_width; k++)
                {
                    output_1d.push_back(output[b][i][j][k]);
                }
            }
        }

    write_to_binary("../outputs/conv3d_nchw_cpp.bin", output_1d);
}

int main() {
    vector<int> input_dims = {1,3,224,224};
    vector<int> kernel_dims = {64,3,7,7};
    auto input = read_npy_file("../inputs/py_input.npy", input_dims);
    auto kernel = read_npy_file("../weights/conv1_wt.npy", kernel_dims);
    // Output tensor
    vector<vector<vector<vector<float>>>> output;

    // // // Perform the convolution
    conv3d(input, kernel, output, 1, 1);

    // // Print the output dimensions and the first value for verification
    cout << "Output shape: [" << output.size() << ", "
         << output[0].size() << ", "
         << output[0][0].size() << ", "
         << output[0][0][0].size() << "]" << endl;
    cout << "First output value: " << output[0][0][0][0] << endl;

    return 0;
}
