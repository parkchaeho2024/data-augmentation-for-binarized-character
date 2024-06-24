# MSTV

## Markovian Stroke Thickness Variation Augmentation for Binarized Character Recognition

This repository contains the Pytorch implementation of [our paper(MSTV)](https://drive.google.com/file/d/1nJxhpwQmrTKYiJb8vOA5QLJo3MsJQxeJ/view?usp=sharing).

![Flowchart of MSTV](/figure/Flowchart.png "This is a sample image.")

## Markov Process Algorithm Summary

The Markov Process algorithm enhances binarized images (background in black, foreground in white) by varying stroke thickness to create diverse image representations. The main steps are:

1. **Boundary Extraction**: Extract boundaries using morphological operations.
2. **Boundary Modification**: Modify the external boundaries of strokes.
3. **Gaussian Smoothing & Binarization**: Apply Gaussian smoothing for natural modification, followed by binarization to obtain the final binarized image.

### Boundary Modification

This augmentation method uses a Markov Process with parameters `pm` (markov probability for width) and `h` (height variation).

1. **Boundary Line Drawing**: Draw lines perpendicular to the stroke boundary by a random value `delta_h` (rand(0~h)).
2. **Probability-Based Modification**: With probability `pm`, decide whether to continue modifying adjacent boundary pixels.
3. **Modification Completion**: If the modification is not continued (1-pm probability), select the next pixel to modify and iterate over all boundary pixels.

This algorithm preserves the original image features while introducing variations, enhancing dataset diversity and improving character recognition models, particularly for ancient document analysis.

### Visualization of augmented data

![Flowchart of MSTV](/figure/Visualization.png "This is a sample image.")



* Green region: the stroke thickness is increased
* Red region: the stroke thickness is decreased
