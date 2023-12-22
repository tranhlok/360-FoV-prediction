360-FoV-prediction
==============================

Team Members
------------
Loc Tran, Taishan Zhao, Yueyu Hu (Mentor)

Abstract
------------
Abstract—This project aims to explore, adapt, and develop a predictive system for anticipating viewers’ gaze directions within immersive images and videos by analyzing the past view direction and, when applicable, the content of the video itself. Additionally, we focus on enhancing the user experience in point cloud video, where six degree-of-freedom is available for viewing volumetric data. By adapting the existing recurrent neural network (RNN) and transformer approach for three degree-of-freedom field-of-view prediction to a new dataset  and exploring potential methodological improvements, our project seeks to improve the immersive quality and engagement of virtual experiences, achieve field-of-view prediction within six degree-of-freedom, with potential applications in fields such as virtual reality gaming, virtual reality live streaming, and educational simulations. The source code implemented in this project can be accessed here. 

Index
------------
Terms—point cloud video, image processing, computer vision, field of view

Dataset
------------
Link to the dataset: https://cuhksz-inml.github.io/user-behavior-in-vv-watching/

Google drive link to directly download the dataset: https://drive.google.com/drive/folders/10IneAJ6uMoI_BI93tuKj5_Oxj9qbkM29

Please download the dataset and put it into the /data/raw folder. The dataset contains 100 text files.

Run the data pre precessing scripts located in src/data. The created dataset structure will be located in the data folder as the following:

    ├── data
    │   ├── processed_by_activity <- The data set for divided by activities
    │   ├── processed      <- The final data
    │   └── raw            <- The original data

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed_by_activity <- The data set for divided by activities
    │   ├── processed      <- The final data
    │   └── raw            <- The original data
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Jupyter notebooks. 
    ├── references         <- Referenced papers, others paper that we have came by.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── data           <- Scripts to precprocess the data
    │   ├── models         <- Scripts to train models and then use trained models to make prediction
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

