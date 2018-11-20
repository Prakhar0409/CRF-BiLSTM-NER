# CRF-BiLSTM-NER

This repository is a NER system for diseases and their treatments. Basically given a set of sentences, each word has to be tagged with either being a disease, a treatment or others.

Labels are be D, T or O signifying disease, treatment or other.

## Model

The model being used in a BiLSTM-CRF (CRF stands for conditional random fields)

Some of the things implemented for improving the model performance are:

* features from lower level syntactic processing like POS tagging
* using existing word embeddings as feature

This has been implemented in PyTorch.
