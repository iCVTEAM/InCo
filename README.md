# Introduction
![Copyright](https://img.shields.io/badge/Copyright-CVTEAM-red)

This code provides an initial version for the implementation of the Neurocomputing paper "Invariant and Consistent: Unsupervised Representation Learning for Few-Shot Visual Recognition''.

[Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S0925231222014692). 


# Prerequisites

Python 3.6+
Pytorch 1.4+
CUDA 10.1


# Training 

python main.py -a convnet --lr 0.03 --batch-size 128 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0   \
 /your dataset path/


# Testing

python eval.py \
  -a convnet  --n_shots 20 --n_queries 5  --n_test_runs 600 \
  --pretrained /model path/ \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /your dataset path/
