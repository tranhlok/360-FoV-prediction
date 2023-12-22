360-FoV-prediction
==============================
*** ON GOING PROJECT, CODEBASE WILL BE UPDATE LATER SINCE WE CONDUCT OUR TRAINING ON GOOGLE COLAB
This project aims to explore, adapt, and develop a predictive system for anticipating viewers' gaze directions within immersive 360 degree images and videos by analyzing the past view direction.

Dataset
------------
Link to the dataset: https://cuhksz-inml.github.io/user-behavior-in-vv-watching/

Google drive link to directly download the dataset: https://drive.google.com/drive/folders/10IneAJ6uMoI_BI93tuKj5_Oxj9qbkM29

    ├── data
    │   ├── processed_by_activity <- The data set for divided by activities
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump

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
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.   
    │
    ├── references         <- Referenced papers, others paper that we have came by.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


