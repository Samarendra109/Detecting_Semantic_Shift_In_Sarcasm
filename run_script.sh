#!/bin/bash
while getopts c:w: flag
do
	case "${flag}" in
		c) context_size=${OPTARG};;
		w) context_weight=${OPTARG};;
	esac
done
python3 glove_transformed.py --c $context_size --w $context_weight
python3 glove_selective_training.py --c $context_size --w $context_weight


