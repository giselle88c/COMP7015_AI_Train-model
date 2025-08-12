# Project Submission

This folder contains the submission for the COMP7015 Group Project (Optional 1).

## Installation

To set up the conda environment for this project, follow the steps below:


1. **Create the conda environment:**

    Install the conda environment using the following command:

    ```bash
    conda env create -f environment.yaml
    ```

**Environment Details:**

- **Python Version:** 3.10.15
- **Package Versions:** 

      - bayesian-optimization==2.0.0
      - imbalanced-learn==0.12.4
      - imblearn==0.0
      - joblib==1.4.2
      - lightgbm==4.5.0
      - matplotlib==3.9.2
      - notebook==7.2.2
      - numpy==1.26.4
      - pandas==2.2.3
      - scipy==1.12.0
      - seaborn==0.13.2
      - tabulate==0.9.0

2. **Activate the conda environment:**

    ```bash
    conda activate comp7015
    ```
3. **Optional:**
- If you would like to run `graph.render('../others/lightgbm/lightgbm_plot_tree', format='jpg')` in `4_train_lightgbm.ipynb` to generate the tree diagram, install and download the graphviz executable from https://graphviz.gitlab.io/download/, and add the `path_to_graphviz\Graphviz\bin` folder to the system PATH, then close and restart all command prompts, Jupyter notebook or VS Code. Uncomment and run that line. 
- The image has been generated for you in `submission/others/lightgbm/lightgbm_plot_tree.jpg`.

## Folder Hierarchy


```
submission/
│
├── codes/
│   ├── 1_Data Pre-processing.ipynb
│   ├── 2_hyperparameter_search.ipynb
│   ├── 3_train_probit_logistic.ipynb
│   ├── 4_train_lightgbm.ipynb
│   ├── 5_make_prediction.ipynb
│   ├── data/
│   │   ├── mimiciv_traindata.csv
│   │   ├── mortality_testdata.csv
│   │   └── option1_test_output_example.csv
│   ├── Models.py # implementation of the probit and logistic model
│   └── util.py # useful functions to compute features, cross validation, and visualize results
│
├── COMP7015 Group Project Report.pdf
├── environment.yaml
│
├── others/
│   ├── lightgbm/ # cross validation result of LightGBM model
│   │   ├── ...
│   ├── lightGBM.json
│   ├── lightgbm.txt
│   ├── lightgbm_pipeline.pkl
│   ├── logistic/ # cross validation result of logistic model
│   │   ├── ....
│   ├── logistic.json
│   ├── logistic.pkl
│   ├── logistic_pipeline.pkl
│   ├── probit/ # cross validation result of probit model
│   │   ├── ...
│   ├── probit.json
│   ├── probit.pkl
│   └── probit_pipeline.pkl
│
├── predictions.csv
└── readme.md
```

## Usage

After setting up the environment, you can run the notebooks in the following orders:

1. **Exploratory data analysis and feature engineering**

- This notebook will give you insight about the data and show you the steps to preprocess the data and compute the features.
```bash
1_Data Pre-processing.ipynb
```
2. **Hyperparameter search**
- This notebook searches for optimal values of hyperparameters for all models (probit, logistic, LightGBM) and save the hyperparameters in the json files of corresponding models (probit.json, logistic.json, lightgbm.json).
```bash
2_hyperparameter_search.ipynb
```
3. **Probit and Logistic models**
- This notebook trains the probit and logistic models using the hyperparameters found in step 2 and all the data, and save the data preprocessing (sklearn) pipeline to `*_pipeline.pkl` as well as the models to `probit.pkl` and `logistic.pkl`. 
- After training, a stratified 5-fold cross validation will be performed to estimate the models' performance, the summaries will be save in the folders `probit` and `logistic` respectively.
```bash
3_train_probit_logistic.ipynb
```

4. **LightGBM model**
- This notebook trains the LightGBM model using the hyperparameters found in step 2 and all the data, and save the data preprocessing (sklearn) pipeline to `lightgbm_pipeline.pkl` as well as the model to `lightgbm.txt`. 
- After training, a stratified 5-fold cross validation will be performed to estimate the model's performance, the summary will be save in the folder `lightgbm`.
```bash
4_train_lightgbm.ipynb
```

5. **Making a prediction**
- This notebook will generate predictions made by the LightGBM model (or also the probit and logistic models).
```bash
5_make_prediction.ipynb
```