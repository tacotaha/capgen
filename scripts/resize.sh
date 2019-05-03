#!/bin/bash

cd ../data/train2014

for i in *.jpg; do
    printf "Resize $i\n"
    convert "$i" -resize 256x256 "$i"
done
