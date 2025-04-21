#!/bin/bash

nTrain=1000

extend=0
stratify=1
k=5

inpath="in/"
outpath="out/"
threads=8

../venv/bin/python pre.py "$nTrain" "$extend" "$stratify" "$k" "$inpath"

NTRAIN=$(grep "n_train:" in/sizes.txt | awk '{print $2}')
NDESC=$(grep "n_desc:" in/sizes.txt | awk '{print $2}')
NTEST=$(grep "n_test:" in/sizes.txt | awk '{print $2}')

echo $NTRAIN
echo $NDESC
echo $NTEST

g++ -I ~/inc/ main.cpp -o krr -fopenmp -DN_TRAIN="$NTRAIN" -DN_DESC="$NDESC" -DN_TEST="$NTEST" -DEIGEN_STACK_ALLOCATION_LIMIT=0

./krr "$k" "$threads" "$inpath" "$outpath"
../../venv/bin/python post.py "$outpath"

