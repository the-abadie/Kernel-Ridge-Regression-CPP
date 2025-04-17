#!/bin/bash

# split.py CL arguments:
# arg1: N_train (int)
# arg2: Extend Data? Yes = 1, No = 0

python ./data/split.py 1000 0
g++ -I lib/ hpc/krr.cpp -o krr_vec -O3

