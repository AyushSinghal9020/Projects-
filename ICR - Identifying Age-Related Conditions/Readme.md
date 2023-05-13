# Goal of the Competition
The goal of this competition is to predict if a person has any of three medical conditions. You are being asked to predict if the person has one or more of any of the three medical conditions $(Class 1)$, or none of the three medical conditions $(Class 0)$. You will create a model trained on measurements of health characteristics.

To determine if someone has these medical conditions requires a long and intrusive process to collect information from patients. With predictive models, we can shorten this process and keep patient details private by collecting key characteristics relative to the conditions, then encoding these characteristics.

# Dataset Description
The competition data comprises over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions -- a binary classification problem.

# Files and Field Descriptions
## Train
|______|______
|---|---
|Id|Unique identifier for each observation.
|AB-GL|Fifty-six anonymized health characteristics. All are numeric except for EJ, which is categorical.
|Class|$1$ indicates the subject has been diagnosed with one of the three conditions
||$0$ indicates they have not.
## Test 
The test set. Your goal is to predict the probability that a subject in this set belongs to each of the two classes.
# Greeks
Supplemental metadata, only available for the training set.
|Alpha|Identifies the type of age-related condition, if present.
||A No age-related condition. Corresponds to class 0.
||B, D, G The three age-related conditions. Correspond to class 1.
|Beta, Gamma, Delta |Three experimental characteristics.
|Epsilon|The date the data for this subject was collected. 
## Sample Submission

|sample_submission.csv |A sample submission file in the correct format. 
