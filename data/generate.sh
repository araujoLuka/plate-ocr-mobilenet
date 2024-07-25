#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

VAL_SOURCE_DIR="/home/lucas/Downloads/placas"
TRAIN_SOURCE_DIR="/home/lucas/Downloads/ocr_train"

BASE_DIR=$SCRIPT_DIR"/../assets/images"
TRAIN_DIR=$BASE_DIR"/train"
NUMBERS_DIR=$TRAIN_DIR"_numbers"
LETTERS_DIR=$TRAIN_DIR"_letters"
VAL_DIR=$BASE_DIR"/val"
DIGIT_SCRIPT=$SCRIPT_DIR"/../utils/digit.py"

TRAIN_COUNT=$((VAL_COUNT + 1))

# Clear previous data
rm -rf $BASE_DIR

mkdir -p $TRAIN_DIR
mkdir -p $VAL_DIR
mkdir -p $NUMBERS_DIR
mkdir -p $LETTERS_DIR

declare -a VAL_IMAGES=($(find $VAL_SOURCE_DIR -type f -name "*.jpg"))
declare -a TRAIN_IMAGES=($(find $TRAIN_SOURCE_DIR -type f -name "*.png"))

echo "Copying validation images..."
cp -r "${VAL_IMAGES[@]}" $VAL_DIR
echo "Finished! Total images: $(ls $VAL_DIR | wc -l)"

# Training images are in sub-dirs
# - There is two datasets: data and data2
# - Each dataset has a sub-dir named 'training_data'
# - Inside 'training_data' there are sub-dirs for each label
# - Only the lastest sub-dirs need to be copied
echo "Copying training images..."

TRAIN_COUNT=0
NUMBERS_COUNT=0
LETTERS_COUNT=0
# For each dataset
for dataset_dir in $(ls -d $TRAIN_SOURCE_DIR/*/)
do
    echo "Processing $dataset_dir..."

    # Access the 'training_data' dir
    training_data_dir=$dataset_dir"training_data/"

    # Store dataset length
    DATASET_COUNT=0

    # For each label
    # - Create a sub-dir in the training dir
    # - Copy all images to the sub-dir
    # - Count them and add to the total
    for label_dir in $(ls -d $training_data_dir*/)
    do
        declare -a IMAGES=($(find $label_dir -type f -name "*.png"))
        TRAIN_COUNT=$((TRAIN_COUNT + ${#IMAGES[@]}))
        DATASET_COUNT=$((DATASET_COUNT + ${#IMAGES[@]}))
        LABEL_DIR=$TRAIN_DIR"/"$(basename $label_dir)
        mkdir -p $LABEL_DIR
        cp -r "${IMAGES[@]}" $LABEL_DIR

        # Check if the label is a number or a letter
        if [[ $(basename $label_dir) =~ ^[0-9]+$ ]]; then
            LABEL_DIR=$NUMBERS_DIR"/"$(basename $label_dir)
            mkdir -p $LABEL_DIR
            cp -r "${IMAGES[@]}" $LABEL_DIR
            NUMBERS_COUNT=$((NUMBERS_COUNT + ${#IMAGES[@]}))
        else
            LABEL_DIR=$LETTERS_DIR"/"$(basename $label_dir)
            mkdir -p $LABEL_DIR
            cp -r "${IMAGES[@]}" $LABEL_DIR
            LETTERS_COUNT=$((LETTERS_COUNT + ${#IMAGES[@]}))
        fi
    done

    echo "Finished! Total images from $dataset_dir: $DATASET_COUNT"
done

echo "Finished!"
echo "> Total training images: $TRAIN_COUNT" "Saved in $TRAIN_DIR"
echo "> Total number images: $NUMBERS_COUNT" "Saved in $NUMBERS_DIR"
echo "> Total letter images: $LETTERS_COUNT" "Saved in $LETTERS_DIR"

echo "Datasets created successfully!"
