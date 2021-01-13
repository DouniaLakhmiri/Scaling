*****
# Scaling deep neural networks
*****

This package is an ready to use blackbox for the [NOMAD](https://www.gerad.ca/nomad/) software. 

## Prerequisites

To run the code, please make sure that the following are correctly installed:

* Python > 3.6
* [PyTorch](https://pytorch.org/)
* GCC > 9.0
* A compiled version of [NOMAD](https://www.gerad.ca/nomad/).


## Getting started

The goal is to scale a baseline network under FLOPS and MACS constraints by optimizing 3 hyperparameters with NOMAD. 
The blackbox provided takes the 3 coefficients as input : 

* d: depth multiplier,
* w: width multiplier
* r: and image resolution

to build a scaled version of the ResNet18() network. The blackbox then trains the new network from scratch and then returns 
the validation accuracy as a measure of performance. 

So far, all tests are done on the CIFAR-10 dataset.

The constrains on the FLOPS and MACS limits these values to the FLOPS and MACS of the baseline network (ResNet18)


### Problem formulation

The optimization problem at hand can be formulated in two ways:

* Formulation 1: Maximize the accuracy with a constraints on the number of FLOPS,
* Formulation 2: Minimize the number of FLOPS with a constraint on the accuracy.


Each formulation is in the corresponding folder with the instructions to launch an optimization with NOMAD.

