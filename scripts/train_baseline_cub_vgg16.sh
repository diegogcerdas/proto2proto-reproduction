#!/bin/sh

nonRandom=org2
runName=student_VGG16
newRun=false
serviceType=recognition
ablationType=VGG19_16_cub

mkdir Experiments/$ablationType/$runName/$nonRandom
expOut=Experiments/$ablationType/$runName/$nonRandom/$nonRandom.out
errorOut=Experiments/$ablationType/$runName/$nonRandom/error+$nonRandom.out

cp Experiments/$ablationType/$runName/args.yaml Experiments/$ablationType/$runName/$nonRandom/args.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py $runName $newRun $serviceType $nonRandom $ablationType > $expOut 2>$errorOut