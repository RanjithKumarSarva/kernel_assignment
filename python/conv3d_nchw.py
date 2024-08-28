from torch import nn
import torch
import numpy as np

def write_binary_file(file_path, array):
    with open(file_path, 'wb') as f:
        array.tofile(f)

def write_npy_file(file_path, array):
    np.save(file_path, array)



if __name__ == "__main__":
    input_dims = [1,3,224,224]
    kernel_dims = [64,3,7,7]

    input_height = input_dims[2]
    input_width = input_dims[3]
    input_channel = input_dims[1]

    kernel_height = kernel_dims[2]
    kernel_width = kernel_dims[3]
    kernel_channel = kernel_dims[1]

    stride = 1
    padding = 1
    output_channel = kernel_dims[0]

    output_height = int((input_height + 2 * padding - kernel_height) / stride + 1)
    output_width = int((input_width + 2 * padding - kernel_width) / stride + 1)
    output_channel = kernel_dims[0]

    conv = nn.Conv2d(
        output_channel,
        input_channel,
        kernel_size=(kernel_height, kernel_height),
        stride=stride,
        padding=padding,
    )
    input_data = np.load("../inputs/py_input.npy")
    input_tensor = torch.from_numpy(input_data)

    kernel_data = np.load("../weights/conv1_wt.npy")
    kernel_tensor = torch.from_numpy(kernel_data)

    temp = 0


    bias_matrix = [(0) for i in range(output_channel)]
    bias_tensor = torch.Tensor(bias_matrix)

    conv.weight.data = kernel_tensor
    conv.bias.data = bias_tensor

    output = conv(input_tensor)
    print("input shape ", input_tensor.shape)
    print("kernel shape ", kernel_tensor.shape)
    print("output shape ", output.shape)
    output_np = output.detach().numpy() 
    output_1d = output_np.flatten()    # Flatten to 1D array

    # Write the 1D array to a binary file
    write_binary_file('../outputs/py_conv3d_nchw_output.bin', output_1d)

    print("3D Convolution complete. Output written to 'py_filter_output.bin'.")