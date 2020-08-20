%starter code for project 2: linear classification
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2017
%Bharadwaj Ravichandran, Jan 2020

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:
    PSU Email ID:
    Description: (A short description of what this script does).
%}

close all;
clear all;
addpath export_fig
% Choose which dataset to use (choices wine, wallpaper, taiji) :
dataset = 'wine';
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);

%% An example Linear Discriminant Classification


% %  Classification here is based on 2 Features (featureA and feature B).  
% %       You will be using all of the features but using 2 features makes it 
% %       easier to visualize than the multidimensional hyperplane
% 
% featureA = 1;
% featureB = 7;
% feature_idx = [featureA,featureB];
% numGroups = length(countcats(test_labels));
% % Uncomment the following line to use all features
% % feature_idx = 1:size(train_featureVector,2);
% 
% 
% train_featureVector = train_featureVector(:,feature_idx);
% test_featureVector = test_featureVector(:,feature_idx);
% 
% %%  Classify the data and show statistics
% %    This example is using Matlab's inbuilt Classifier.
% %        You do not need to follow their data management style, this is
% %        to allow you to see the result and show some visualizations
% %        of the classification
% % Train the model (you will have to write this function)
% MdlLinear = fitcdiscr(train_featureVector,train_labels)
% 
% % Find the training accurracy (you will have to write testing 
% %      function (the function for finding the class labels from a set of
% %      features)
% train_pred = predict(MdlLinear,train_featureVector);
% % Create confusion matrix
% train_ConfMat = confusionmat(train_labels,train_pred)
% % Create classification matrix (rows should sum to 1)
% train_ClassMat = train_ConfMat./(meshgrid(countcats(train_labels))')
% % mean group accuracy and std
% train_acc = mean(diag(train_ClassMat))
% train_std = std(diag(train_ClassMat))
% 
% % Find the testing accurracy (you will have to write testing 
% %      function (the function for finding the class labels from a set of
% %      features)
% test_pred = predict(MdlLinear,test_featureVector);
% % Create confusion matrix
% test_ConfMat = confusionmat(test_labels,test_pred)
% % Create classification matrix (rows should sum to 1)
% test_ClassMat = test_ConfMat./(meshgrid(countcats(test_labels))')
% % mean group accuracy and std
% test_acc = mean(diag(test_ClassMat))
% test_std = std(diag(test_ClassMat))
% 
% 
% 
% 
% % %%  Display the linear discriminants and a set of features in two of the feature dimensions
% % %      You will need to modify this function for to use your LDA
% % %      classification boundries to work with your code.. Look at the code
% % %      for more details
% figure(1)
% visualizeBoundaries(MdlLinear,test_featureVector,test_labels,1,2)
% title('{\bf Linear Discriminant Classification}')
% export_fig linear_discriminant_example -png -transparent
% %%  Display the classified regions of two of the feature dimensions  
% %      You will need to modify this function for with your testing 
% %      function (the function for finding the class labels from a set of
% %      features).
% figure(2)
% h = visualizeBoundariesFill(MdlLinear,test_featureVector,test_labels,1,2);
% title('{\bf Classification Area}')
% export_fig classification_fill_example -png -transparent


%% LDA Wine dataset
% Choose which dataset to use (choices wine, wallpaper, taiji) :
disp("**********************************Wine**************************************");
wi_dataset = 'wine';

[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(wi_dataset);
temp_train_featureVector = train_featureVector;
temp_test_featureVector = test_featureVector;

featureA = 1;
featureB = 7;
feature_idx = [featureA,featureB];
numGroups = length(countcats(test_labels));
% Uncomment the following line to use all features
% feature_idx = 1:size(train_featureVector,2);

train_featureVector = train_featureVector(:,feature_idx);
test_featureVector = test_featureVector(:,feature_idx);

W_star = train_LDA(train_featureVector, train_labels, numGroups)

train_pred = predict_LDA(W_star, train_featureVector);

% Create confusion matrix
train_ConfMat = confusionmat(train_labels,categorical(train_pred))
% Create classification matrix (rows should sum to 1)
train_ClassMat = train_ConfMat./(meshgrid(countcats(train_labels))')
% mean group accuracy and std
train_acc = mean(diag(train_ClassMat))
train_std = std(diag(train_ClassMat))

figure(1);
cm = confusionchart(train_ConfMat);
st = " Training Conf Matrix("+wi_dataset+"), Acu = " + string(train_acc*100) + "%, SD = " + string(train_std);
cm.Title =  st;
export_fig LDA_wine_2_feature_train -png -transparent

% Find the testing accurracy (you will have to write testing 
%      function (the function for finding the class labels from a set of
%      features)
test_pred = predict_LDA(W_star, test_featureVector);
% Create confusion matrix
test_ConfMat = confusionmat(test_labels,categorical(test_pred))
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(test_labels))')
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat))
test_std = std(diag(test_ClassMat))

figure(2);
cm = confusionchart(test_ConfMat);
st = " Testing Conf Matrix("+wi_dataset+"), Acu = " + string(test_acc*100) + "%, SD = " + string(test_std);
cm.Title =  st;
export_fig LDA_wine_2_feature_test -png -transparent


visualizeBoundaries1(W_star,test_featureVector,test_labels,1,2)
title('{\bf Linear Discriminant Classification with 2 feature}')
export_fig LDA_2feature_bountries -png -transparent
%%  Display the classified regions of two of the feature dimensions  
%      You will need to modify this function for with your testing 
%      function (the function for finding the class labels from a set of
%      features).
% figure(4)
% h = visualizeBoundariesFill(W_star,test_featureVector,test_labels,1,2);
% title('{\bf Classification Area}')
% export_fig classification_fill_example -png -transparent

%% LDA Wine dataset, All feature

train_featureVector = temp_train_featureVector;
test_featureVector = temp_test_featureVector;
numGroups = length(countcats(test_labels));
% Uncomment the following line to use all features
feature_idx = 1:size(train_featureVector,2);

train_featureVector = train_featureVector(:,feature_idx);
test_featureVector = test_featureVector(:,feature_idx);

W_star = train_LDA(train_featureVector, train_labels, numGroups)

train_pred = predict_LDA(W_star, train_featureVector);

% Create confusion matrix
train_ConfMat = confusionmat(train_labels,categorical(train_pred))
% Create classification matrix (rows should sum to 1)
train_ClassMat = train_ConfMat./(meshgrid(countcats(train_labels))')
% mean group accuracy and std
train_acc = mean(diag(train_ClassMat))
train_std = std(diag(train_ClassMat))

figure(5);
cm = confusionchart(train_ConfMat);
st = " Training Conf Matrix("+wi_dataset+"), Acu = " + string(train_acc*100) + "%, SD = " + string(train_std);
cm.Title =  st;
export_fig LDA_wine_all_feature_train -png -transparent

% Find the testing accurracy (you will have to write testing 
%      function (the function for finding the class labels from a set of
%      features)
test_pred = predict_LDA(W_star, test_featureVector);
% Create confusion matrix
test_ConfMat = confusionmat(test_labels,categorical(test_pred))
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(test_labels))')
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat))
test_std = std(diag(test_ClassMat))


figure(6);
cm = confusionchart(test_ConfMat);
st = " Testing Conf Matrix("+wi_dataset+"), Acu = " + string(test_acc*100) + "%, SD = " + string(test_std);
cm.Title =  st;
export_fig LDA_wine_all_feature_test -png -transparent


% plotConfMat(test_ConfMat);