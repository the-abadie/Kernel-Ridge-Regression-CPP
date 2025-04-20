#!/bin/bash

nTrain=1000
extend=0
k=5
inpath=in/
outpath=out/

threads=20

# pre.py CL arguments:
# arg1: N_train (int)
# arg2: Extend Data? (Yes = 1, No = 0)
# arg3: k (number of folds, int)
# arg4: path to input data folder (include /)
../venv/bin/python pre.py $nTrain $extend $inpath

# krr CL arguments:
# arg1: path to input  data folder (include /)
# arg2: path to output data folder (include /)
./krr $inpath $outpath

../venv/bin/python post.py $outpath
