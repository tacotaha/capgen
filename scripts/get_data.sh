#!/usr/bin/env bash
#title          :get_data:.sh
#description    :This script downloads the MOCO training dataset 
#author         :Taha Azzaoui <tazzaoui@cs.uml.edu>
#version        :1    
#usage          :./get_data.sh
#================================================================

mkdir -p ../model
mkdir -p ../data
cd ../data

echo -e "Downloading Training Images...\n"
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2017.zip 

echo -e "Downloading Annotations...\n"
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip


echo -e "Downloading Validation Images...\n"
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
