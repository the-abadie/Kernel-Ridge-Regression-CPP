#!/bin/bash

nTrain=1000
extend=0
stratify=1
k=5
inpath="in/"
outpath="out/"
threads=8

python pre.py "$nTrain" "$extend" "$stratify" "$k" "$inpath"
./krr "$k" "$threads" "$inpath" "$outpath"
python post.py "$outpath"

