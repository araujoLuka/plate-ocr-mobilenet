#!/bin/bash

# Diretório contendo as imagens das placas
SOURCE_DIR="/home/lucas/Downloads/placas/"
BASE_DIR="../assets/images"
TRAIN_DIR=$BASE_DIR"/train/"
NUMBERS_DIR=$TRAIN_DIR"numbers/"
LETTERS_DIR=$TRAIN_DIR"letters/"
VAL_DIR=$BASE_DIR"/val"
DIGIT_SCRIPT="../utils/digit.py"

VAL_COUNT=100
TRAIN_COUNT=$((VAL_COUNT * 3))

# Cria os diretórios de treino e avaliação se não existirem
mkdir -p $TRAIN_DIR
mkdir -p $VAL_DIR

# Lista todas as imagens no diretório de origem
declare -a IMAGES=($(find $SOURCE_DIR -type f -name "*.jpg"))

# Embaralha a lista de imagens
shuf -n ${#IMAGES[@]} -e "${IMAGES[@]}" -o shuffled_images.txt

# Pega os primeiros VAL_COUNT arquivos para avaliação e o restante para treino
head -n $VAL_COUNT shuffled_images.txt > val_images.txt
tail -n $TRAIN_COUNT shuffled_images.txt > train_images.txt

copy_images() {
    local image_list=$1
    local output_dir=$2
    while IFS= read -r image; do
        cp "$image" $output_dir
    done < "$image_list"
}

# Função para processar as imagens usando digit.py
process_images() {
    local image_list=$1
    local output_dir=$2
    local extra_args=${3:-""}

    while IFS= read -r image; do
        if [ ! -z "$extra_args" ]; then
            python3 $DIGIT_SCRIPT --save-dir "$output_dir" --all "$extra_args" "$image"
        else
            python3 $DIGIT_SCRIPT --save-dir "$output_dir" --all "$image"
        fi
    done < "$image_list"
}

# Processa as imagens de treino
echo "Processing training images..."
process_images "train_images.txt" $TRAIN_DIR
echo "Finished! Total images: $(ls $TRAIN_DIR | wc -l)"

# Processa numeros e letras separadamente
echo "Processing training images (numbers only)..."
process_images "train_images.txt" $NUMBERS_DIR "--number-only"
echo "Finished! Total images: $(ls $NUMBERS_DIR | wc -l)"

echo "Processing training images (letters only)..."
process_images "train_images.txt" $LETTERS_DIR "--letter-only"
echo "Finished! Total images: $(ls $LETTERS_DIR | wc -l)"

# Processa as imagens de avaliação
echo "Processing validation images..."
copy_images "val_images.txt" $VAL_DIR
echo "Finished! Total images: $(ls $VAL_DIR | wc -l)"

# Limpa os arquivos temporários
rm shuffled_images.txt train_images.txt val_images.txt

echo "Datasets created successfully!"
