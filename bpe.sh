#!/bin/bash -v

# suffix of source language files
SRC=uy
TRG=ch
STM=stem

# number of merge operations
bpe_operations=32000

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=subword-nmt/subword_nmt

# train BPE
cat corpus/train.$SRC | ${subword_nmt}/learn_bpe.py -s ${bpe_operations} > $SRC.bpe
cat corpus/train.$TRG | ${subword_nmt}/learn_bpe.py -s ${bpe_operations} > $TRG.bpe
cat corpus/train.$STM | ${subword_nmt}/learn_bpe.py -s ${bpe_operations} > $STM.bpe

# apply BPE
for prefix in train val test test2
do
    ${subword_nmt}/apply_bpe.py -c $SRC.bpe < corpus/$prefix.$SRC > corpus/$prefix.bpe.$SRC
    ${subword_nmt}/apply_bpe.py -c $TRG.bpe < corpus/$prefix.$TRG > corpus/$prefix.bpe.$TRG
    ${subword_nmt}/apply_bpe.py -c $STM.bpe < corpus/$prefix.$STM > corpus/$prefix.bpe.$STM
done