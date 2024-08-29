# Kernel Writing
The repo contains the Independent C++ inference of different Conv kernels and its verification python files

## Machine Requirements:
- Processor Architecture: x86_64
- RAM: Minimum 8GB
- OS: Ubuntu 20.04 
- Storage: Minimum 64GB

## Prequisites
* G++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
* cmake version 3.29.3
* GNU Make 4.2.1
* [cnpy](https://github.com/rogersce/cnpy)
* Python 3.8.10 (create venv)

## Install Prequisites
1. Build the cnpy library by following the steps in [documentation](https://github.com/rogersce/cnpy?tab=readme-ov-file#installation)  

## Cloning the repo
Use the command below to clone the repo
```
    git clone https://github.com/RanjithKumarSarva/kernel_assignment.git
    cd kernel_practice
```
## Activate python environment
```
source path_to_python_env/bin/activate
pip install -r kernel_practice/requirements.txt
```

## How to Build and Run Conv3d with NCHW data format

```
    source run_conv3d_nchw
```
1. Input image shape [1x3x224x224], Weights shape [64x3x7x7]
2. expected output 
```
Output shape: [1, 64, 220, 220]
First output value: 0.542983
Python computation
        3D Convolution complete. Output written to 'py_conv3d_nchw_output.bin'.
        3D Convolution complete. Output written to 'py_conv3d_nhwc_output.bin'.
Min difference:  0.0
Max difference:  5.4836273e-06
Mean difference:  2.3758271e-07
Files are identical
```

## How to Build and Run Conv3d with NHWC data format

```
    source run_conv3d_nhwc
```
1. Input image shape [1x224x224x3], Weights shape [7x7x3x64]
2. expected output 
```
Output shape: [1, 220, 220, 64]
First output value: 0.542983
Python computation
        3D Convolution complete. Output written to 'py_conv3d_nchw_output.bin'.
        3D Convolution complete. Output written to 'py_conv3d_nhwc_output.bin'.
Min difference:  0.0
Max difference:  0.0
Mean difference:  0.0
Files are identical
```