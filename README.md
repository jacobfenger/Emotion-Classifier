# FaceEmotions
Repository for the classification of emotion from images of faces.

### Data can be found on Kaggle:

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

### Steps to prepare:

1. Extract the data (I used tar -xvjf 'filename') into the data/fer2013/ folder
2. Run extract.py (You may have to change a filename or path in the script)
3. Data should be split up into train/public_test/private_test sets

## Scripts in the Scripts Folder

Extract_image.py: pulls the first sample out of the training set and saves it as a .jpg image 

Inspect_data.py: this just counts the categories for both the training/testing set. It will also graph the distribution in a bar graph.
