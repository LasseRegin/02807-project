#!/bin/sh

# Run serial version
echo "Serial version"
PYTHONHASHSEED=1 time python3 train.py

echo ""

echo "Parallel version"
# Run parallel version
PYTHONHASHSEED=1 time python3 train_para.py
