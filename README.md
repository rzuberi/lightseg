# lightseg

TODOS:
- test cellmatching with 64 channels from CPs encFeats and 64 channels from Masks encFeats
- maybe speed up the cellmatching (3mins for 2 images)
- train a better cell segmentation model by first training a better cellpose model with the human-in-the-loop training
- get the groundtruth indexes written, maybe develop an easy way to give the ones that are correct or incorrect with keyboard arrows

A cell nuclei instance segmentation Machine Learning model distilled from CellPose.
This goal of this repository is to be a lighter version of CellPose.
Works as a double U-Net:
- first predicts cell probabilites;
- second predicts binary cell mask;
- contour algorithm generates instances.

The demo.py file gives a usage demonstration.
