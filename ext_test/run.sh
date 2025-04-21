#!/bin/bash

nTrain=1000
extend=1
stratify=1
k=5
inpath=in/
outpath=out/

threads=5

# pre.py CL arguments:
# arg1: N_train (int)
# arg2: Extend Data? (Yes = 1, No = 0)
# arg3: Prefold and stratify data? (Yes = 1, No = 0)
# arg4: k (number of folds, int)
# arg5: path to input data folder (include /)
../venv/bin/python pre.py $nTrain $extend $stratify $k $inpath

# krr CL arguments:
# arg1: k (number of folds, int)
# arg2: number of threads (int)
# arg3: path to input  data folder (include /)
# arg4: path to output data folder (include /)
./krr $k $threads $inpath $outpath

../venv/bin/python post.py $outpath
