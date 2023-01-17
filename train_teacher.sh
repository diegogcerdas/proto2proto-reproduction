#!/bin/sh

RANDOMN=$$
RANDOM=org
nonRandom=org2
runName=teacher_Resnet50
newRun=false
serviceType=recognition
ablationType=Resnet50_18_cars

mkdir Experiments/$ablationType/$runName/$nonRandom
expOut=Experiments/$ablationType/$runName/$nonRandom/$nonRandom.out
errorOut=Experiments/$ablationType/$runName/$nonRandom/error+$nonRandom.out

cp Experiments/$ablationType/$runName/args.yaml Experiments/$ablationType/$runName/$nonRandom/args.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py $runName $newRun $serviceType $nonRandom $ablationType > $expOut 2>$errorOut
