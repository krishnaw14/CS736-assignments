# CS736-assignments
Assignments for the Course CS 736: Medical Image Computing

## Assignment 1: Shape Analysis

Shape Analysis using correspondance based pointset regime on ellipse, leaf, human hand and brain mri image dataset. The pointset was manually developed as part of the assignment. Computation of pre-shape space, alignment of pointsets in pre-shape space, mean shape computation and modes of variation were computed and analyzed.

## Assignment 2: Image Denoising with Markov Random Fields (MRFs)

Implemented a maximum-a-posteriori bayesian denoising algorithm for images on brain mri (grayscale) and digital histology tissue (coloured) images. A MRF based prior (with 4 neighbourhood system) was used for the images and three different potential functions (quadratic, dicontinuity adaptive huber and discontiniuity adaptive) defined on the graph cliques were analysed. Gradient ascent using dynamic step-size was used to optimize the objective function (log of posterior). Likelihood and prior weightage were tuned for different images. 
