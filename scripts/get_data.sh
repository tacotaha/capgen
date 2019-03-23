#!/usr/bin/env bash
#title          :get_data:.sh
#description    :This script downloads the MOCO training dataset 
#author         :Taha Azzaoui <tazzaoui@cs.uml.edu>
#version        :1    
#usage          :./get_data.sh
#=================================================================================

mkdir -p ../data
cd ../data

echo -e "Downloading Images...\n"
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip 

echo -e "Downloading Annotations...\n"
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
