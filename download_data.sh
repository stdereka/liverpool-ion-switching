#!/usr/bin/env bash

echo "Downloading competition data..."
kaggle datasets download -d cdeotte/data-without-drift
echo "Done"

echo "Extracting files..."
unzip data-without-drift.zip
mkdir data/
mkdir data/raw/
mkdir data/processed/
mv test_clean.csv data/raw/
mv train_clean.csv data/raw/
rm data-without-drift.zip
echo "Done"

echo "Downloading pretrained models..."
wget https://www.dropbox.com/s/utfjg92hqsi5mr9/models.zip
echo "Done"

echo "Extracting files..."
unzip models.zip
echo "Done"
