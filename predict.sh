#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "Usage: ./predict.sh [input file] [output file]"
    exit 1
fi

if [ -f $2 ]; then
    echo "Output file already exists. Overwrite this file? (Enter 'yes' to overwrite, otherwise the program will exit)"
    read RESPONSE
    if [ "$RESPONSE" != "yes" ]; then
        echo "Exiting..."
        exit 0
    fi
fi

OUTPUT_DIR=$(dirname $2)

docker run -it --mount type=bind,src=$1,dst=$1 \
    --mount type=bind,src=$OUTPUT_DIR,dst=$OUTPUT_DIR \
    blevine0001/k12ai python predict_k12.py $1 $2
