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
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int in_channels = input[0][0][0].size();

    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    int out_channels = kernel[0][0][0].size();

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    // Initialize the output tensor
    output.assign(batch_size, vector<vector<vector<float>>>(
                                  output_height, vector<vector<float>>(
                                                  output_width, vector<float>(out_channels, 0))));


    vector<vector<vector<vector<float>>>> padded_input(batch_size, vector<vector<vector<float>>>(
            input_height + 2 * padding, vector<vector<float>>(
            input_width + 2 * padding, vector<float>(in_channels, 0))));

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_height; ++i) {
            for (int j = 0; j < input_width; ++j) {
                for (int c = 0; c < in_channels; ++c) {
                    padded_input[b][i + padding][j + padding][c] = input[b][i][j][c];
                }
            }
        }
    }
    // // Perform the convolution operation
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    float sum = 0.0;
                    std::cout<<"["<<b<<"]"<<"["<<i<<"]"<<"["<<j<<"]"<<"["<<oc<<"]"<<std::endl;
                    std::cout<<"kernelstart"<<std::endl;
                    for (int ki = 0; ki < kernel_height; ++ki) {
                        for (int kj = 0; kj < kernel_width; ++kj) {
                            for (int ic = 0; ic < in_channels; ++ic) {
                                // std::cout<<"["<<ki<<"]"<<"["<<kj<<"]"<<"["<<ic<<"]"<<std::endl;
                                int input_i = i * stride + ki;
                                int input_j = j * stride + kj;
                                std::cout<<"["<<b<<"]"<<"["<<input_i<<"]"<<"["<<input_j<<"]"<<"["<<ic<<"]"<<std::endl;
                                sum += padded_input[b][input_i][input_j][ic] * kernel[ki][kj][ic][oc];
                            }
                        }
                    }
                    output[b][i][j][oc] = sum;
                    // break;
                }
            }
        }
    }
}

int main() {
    // input tensor: [batch_size, height, width, channels= 1x64x64x3]
    vector<vector<vector<vector<float>>>> input(1, vector<vector<vector<float>>>(
                                                      6, vector<vector<float>>(
                                                          6, vector<float>(3, 1.0))));

    // kernel [ kernel_height, kernel_width,in_channels, out_channels= 3x3x3x64]
    vector<vector<vector<vector<float>>>> kernel(3, vector<vector<vector<float>>>(
                                                        3, vector<vector<float>>(
                                                            3, vector<float>(2, 0.5))));
    
    auto kernel_data = read_npy_file("/home/ubuntu/acl_resnet18_inference-main/dnnl_resnet18_inference/inputs/py_input.npy");

    // Output tensor
    vector<vector<vector<vector<float>>>> output;

    // // // Perform the convolution
    // conv3d(input, kernel, output, 1, 1);

    // // // Print the output dimensions and the first value for verification
    // cout << "Output shape: [" << output.size() << ", "
    //      << output[0].size() << ", "
    //      << output[0][0].size() << ", "
    //      << output[0][0][0].size() << "]" << endl;
    // cout << "First output value: " << output[0][0][0][0] << endl;

    return 0;
}
