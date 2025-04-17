#!/bin/bash

g++ -I include/ krr.cpp -o krr -O3

# split.py CL arguments:
# arg1: N_train (int)
# arg2: Extend Data? Yes = 1, No = 0
# arg3: path to input  data folder (include /)
python ./data/split.py 1000 0

# krr CL arguments:
# arg1: path to input  data folder (include /)
# arg2: path to output data folder (include /)
./test