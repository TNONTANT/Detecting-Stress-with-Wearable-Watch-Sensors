# FOLLOW INSTRUCTION BELOW TO GET THE RESULT AS REPORT

1. Download raw dataset from this link: https://datadryad.org/stash/dataset/doi:10.5061/dryad.5hqbzkh6f

2. To Pre-processing the raw signal go to pre-processing directory and follow these step.
- run combine.py to get combine signal -> delivarable: csv file of each signal from all user.
- run merge.py to get all signal merged together -> deliverable: single csv file contain all signal combined.
- run label.py to label merged data -> deliverable: single csv file contain all signal combined and labeled stress column.

3. `function.py` 
- Contains function below to be use in feature extraction, Model training and evaluation.
    - stat_feature
    - feature_extraction
    - Model_train
    - eval_testset
    - SearchCV
    - Model_performance_bar

*All instruction to use code has been commented inside function

4. `notebook.ipynb`: Contain all content below
- Exploratory data analysis (EDA).
- Feature extraction implementation.
- Model train and test on original training set and SMOTE training set
- Show all result of training model in figure including classification report

