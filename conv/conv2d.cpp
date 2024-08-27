#include <iostream>
#include <vector>

using namespace std;

// Define the convolution function
void conv3d(const vector<vector<vector<vector<float>>>> &input,
            const vector<vector<vector<vector<float>>>> &kernel,
            vector<vector<vector<vector<float>>>> &output) {

    int batch_size = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int in_channels = input[0][0][0].size();

    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    int out_channels = kernel[0][0][0].size();

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    // Initialize the output tensor
    output.assign(batch_size, vector<vector<vector<float>>>(
                                  output_height, vector<vector<float>>(
                                                  output_width, vector<float>(out_channels, 0))));


    // // Perform the convolution operation
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    float sum = 0.0;
                    for (int ki = 0; ki < kernel_height; ++ki) {
                        for (int kj = 0; kj < kernel_width; ++kj) {
                            for (int ic = 0; ic < in_channels; ++ic) {
                                int input_i = i + ki;
                                int input_j = j + kj;
                                sum += input[b][input_i][input_j][ic] * kernel[ki][kj][ic][oc];
                            }
                        }
                    }
                    output[b][i][j][oc] = sum;
                }
            }
        }
    }
}

int main() {
    // input tensor: [batch_size, height, width, channels= 1x1x6x3]
    vector<vector<vector<vector<float>>>> input(1, vector<vector<vector<float>>>(
                                                      1, vector<vector<float>>(
                                                          6, vector<float>(3, 1.0))));

    // kernel [ kernel_height, kernel_width,in_channels, out_channels= 1x1x3x2]
    vector<vector<vector<vector<float>>>> kernel(1, vector<vector<vector<float>>>(
                                                        1, vector<vector<float>>(
                                                            3, vector<float>(2, 0.5))));

    // Output tensor
    vector<vector<vector<vector<float>>>> output;

    // // Perform the convolution
    conv3d(input, kernel, output);

    // // Print the output dimensions and the first value for verification
    cout << "Output shape: [" << output.size() << ", "
         << output[0].size() << ", "
         << output[0][0].size() << ", "
         << output[0][0][0].size() << "]" << endl;
    cout << "First output value: " << output[0][0][0][0] << endl;

    return 0;
}
