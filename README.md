# Code Repository for Master's Thesis
K-POP: Predicting Distance to Focal Plane for Kato-Katz Prepared Microscopy Slides Using Deep Learning

This repository contains all code produced to write my master's thesis available [here](https://github.com/13hannes11/master_thesis).

The model code which is integrated as a git submodule is hosted on [HuggingFace](https://huggingface.co/13hannes11/master_thesis_models).

The machine learning pipeline works as follows:

1. Using the code in `annotation-preprocessing` metadata and images are extracted from a database. The data is prepared to be usable by the tool Focus Annotator.
2. Data is annotated using Focus Annotator (folder: `focus_annotator`)
3. Annotations are post processed to csv format to be easier to load with pandas (folder: `data-preprocessing`).
4. The models are trained and evaluated in `models`.
