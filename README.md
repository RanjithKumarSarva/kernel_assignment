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
input shape  torch.Size([1, 3, 224, 224])
kernel shape  torch.Size([64, 3, 7, 7])
output shape  torch.Size([1, 64, 220, 220])
3D Convolution complete. Output written to 'py_filter_output.bin'.
Min difference:  0.0
Max difference:  5.4836273e-06
Mean difference:  2.3758271e-07
Files are identical
```