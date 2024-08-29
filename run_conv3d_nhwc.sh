#!/bin/bash
cd conv_with_pad
# Compile the C++ code
g++ conv3d_nhwc.cpp ../utils/utils.cpp -o conv3d_nhwc -lcnpy

# Execute the compiled program
./conv3d_nhwc

cd ../python
# Run the Python script for convolution
python conv3d.py

# Compare the output binary files
python validate.py ../outputs/conv3d_nhwc_cpp.bin ../outputs/py_conv3d_nhwc_output.bin
cd ..
