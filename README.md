# Plant Seedlings Classification

This project is aimed to explore techniques for detecting and distinguishing weeds among the variety of crop seedlings.
The considered dataset is a part of the database that has been recorded at the Aarhus University Flakkebjerg Research station in the collaboration between the University of Southern Denmark and Aarhus University. Images are available to researchers at https://vision.eng.au.dk/plant-seedlings-dataset/


## Table of Contents 

- [Project structure](#Project structure)
- [Data processing](#Data processing)
- [Classification](#Classification)
- [Results](#Results)

## Project structure
The project is organized as follows:

    ├── data
    │   ├── processed              <- The final, canonical data sets for modeling.
    │   └── raw                    <- The original, immutable data dump.
    │
    │
    ├── models                     <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                  <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                 the creator's initials, and a short `-` delimited description, e.g.
    │                                 `1.0-jqp-initial-data-exploration`.
    │
    ├── reports                    <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── paper                  <- Paper based on the work
    │   └── report                 <- Summer practice report 
    │
    ├── requirements.txt           <- The requirements file for reproducing the analysis environment, e.g.
    |
    ├── src                        <- Source code for use in this project.
    │   ├── __init__.py            <- Makes src a Python module
    │   │
    │   ├── data                   <- Scripts to download or generate data
    │   │   └── make_dataset.py    <- Contains functions for images reading, resizing and features extracting. 
                                      Can be runned as script (resizes images from data/raw and writes them at data/processed)
    │   │
    │   ├── features               <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py  <- Contains functions for calculationg contours features
    │   │
    │   └── visualization          <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py       <- Contains functions for segmentation and plotting images

## Data processing

Data processing includes the following steps:
1.  Resolution reducing
2.  Segmentation
3.  Denoising
4.  Feature selection and extraction

The features used in the task can be united in groups:
1. Shape features
2. Colour features

## Classification

The goal of this work is to reach good classification quality for lower processing time and computational complexity. It was decided to use basic classification algorithms in their pure form:
1. Support Vector Machine
2. k-Nearest Neighbours
3. Naive Bayes
4. Decision Tree

## Results
The best reached with SVM; the Micro-averaged F-score metric is 0.89. Detailed results can be found in the paper located at reports/paper/paper.pdf

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
