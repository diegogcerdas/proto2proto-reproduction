#!/bin/sh

nonRandom=org
runName=teacher_Resnet50
newRun=false
serviceType=recognition
ablationType=Resnet50_18_cub

mkdir Experiments/$ablationType/$runName/$nonRandom
expOut=Experiments/$ablationType/$runName/$nonRandom/$nonRandom.out
errorOut=Experiments/$ablationType/$runName/$nonRandom/error+$nonRandom.out

cp Experiments/$ablationType/$runName/args.yaml Experiments/$ablationType/$runName/$nonRandom/args.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py $runName $newRun $serviceType $nonRandom $ablationType > $expOut 2>$errorOut