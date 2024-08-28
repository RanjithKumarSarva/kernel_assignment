#!/bin/bash
cd conv_with_pad
# Compile the C++ code
g++ conv3d_nchw.cpp ../utils/utils.cpp -o conv3d_nchw -lcnpy

# Execute the compiled program
./conv3d_nchw

cd ../python
# Run the Python script for convolution
python conv3d_nchw.py

# Compare the output binary files
python validate.py ../outputs/conv3d_nchw_cpp.bin ../outputs/py_conv3d_nchw_output.bin
cd ..
