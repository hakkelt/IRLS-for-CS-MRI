#!/bin/sh

for f in $1/*
do
    nohup ./run.sh ${f%.*} > ${f%.*}.log &
done