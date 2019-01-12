#!/bin/bash

for i in {0..9}
do
	python2 src/run.py -c src/rna_config.yaml -e train -ts data/${i}_src.train -td data/${i}_trg.train -vs data/${i}_src.train -vd data/${i}_trg.train
	python2 src/run.py -c src/rna_config.yaml -e test -tes data/${i}_src.test -ted data/${i}_trg.test -teo outs/${i}_test.out
done
