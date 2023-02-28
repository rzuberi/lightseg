# lightseg

A cell nuclei instance segmentation Machine Learning model distilled from CellPose.
This goal of this repository is to be a lighter version of CellPose.
Works as a double U-Net:
- first predicts cell probabilites;
- second predicts binary cell mask;
- contour algorithm generates instances.

The demo.py file gives a usage demonstration.
