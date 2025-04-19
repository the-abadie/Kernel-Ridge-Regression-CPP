#!/bin/bash

# pre.py CL arguments:
# arg1: N_train (int)
# arg2: Extend Data? Yes = 1, No = 0
# arg3: path to input data folder (include /)
../venv/bin/python pre.py 1000 1 in/

# krr CL arguments:
# arg1: path to input  data folder (include /)
# arg2: path to output data folder (include /)
./krr in/ out/

../venv/bin/python post.py out/
