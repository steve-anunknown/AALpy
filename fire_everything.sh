#!/bin/bash

if [ ! -f mealy_conformance_testing.py ]
then
    echo "mealy_conformance_testing.py not found"
    exit 1
fi

if [ ! -f make_plots.py ]
then
    echo "make_plots.py not found"
    exit 1
fi

if [ ! -d results ]
then
    echo "results directory not found"
    exit 1
fi

if [ ! -d ~/PythonEnvs/ModelLearning ]
then
    echo "ModelLearning virtual environment not found"
    exit 1
fi

for method in state_coverage rwpmethod wpmethod wmethod
do
    pypy mealy_conformance_testing.py -p -s -f all -b $method
done

source ~/PythonEnvs/ModelLearning/bin/activate

for method in state_coverage rwpmethod wpmethod wmethod
do
    for mode in all combined
    do
        python make_plots.py -r results -b $method -p $mode
    done
done

