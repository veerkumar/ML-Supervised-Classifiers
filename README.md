# Supervised-Classifiers


Implementation:
The functions implemented are:

    A function to train a linear discriminate using least squares for classification .   function takes the training features and labels from the dataset and returns the linear discriminant functions for a 'one-vs-one' or a 'one-vs-all' scheme.
    A function that takes linear discriminant functions (from the previous function) and a set of features to test and returns class labels found with classifier features.
    A function to find the Fisher projection using the training features and labels and also train a classifier to the Fisher projected training data.  The classifier can be a KNN classifier or from Decision Theory using an optimum threshold.  This function returns the Fisher projection coefficients and the corresponding fitted classifier necessary for the testing function.
    A function which takes the output of the previous function  (the Fisher projection and the classifier) and a set of features to test and returns the class labels of the features found by your classifier.

Later quantitatively evaluate BOTH of the classification methods above using the THREE datasets given.

Dataset:
1) WINE  (http://archive.ics.uci.edu/ml/datasets/Wine (Links to an external site.)) dataset.
2) Wallpaper Group Dataset - This dataset consists of the features extracted from images containing the 17 Wallpaper Groups
3) Taiji Pose Dataset - This is a dataset of the joint angles (in quaternions) of 35 sequences from 4 people performing Taiji in our motion capture lab.



# This project contains: 
- `start.m`: the main function which contains examples of: 
   - how to load the data 
   - a classification using LDA with Matlab's inbuilt tools
   - visualizations of the classification
- `loadDataset.m`:  a function to load the specific dataset
- `visualizeBoundaries.m`: a function to visualize the linear discriminant boundaries in two dimensions
- `visualizeBoundariesFill.m`: a function to visualize the classification areas in two dimensions
