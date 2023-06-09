# Dataset Description
The dataset for this competition (both train and test) was generated from a deep learning model trained on the **[Crab Age Prediction](https://www.kaggle.com/datasets/sidhus/crab-age-prediction)** dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Note: You can use this **[notebook](https://www.kaggle.com/code/inversion/make-synthetic-crab-age-data/notebook)** to generate additional synthetic data for this competition if you would like.
****
# Files
* **train.csv** - the training dataset; `Age` is the target
* **test.csv** - the test dataset; your objective is to predict the probability of `Age` (the ground truth is `int` but you can predict `int` or `float`)
* **sample_submission.csv** - a sample submission file in the correct format
