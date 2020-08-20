%starter code for project 2: linear classification
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2017
%Bharadwaj Ravichandran, Jan 2020

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Pramod Kumar
    PSU Email ID: pjk5502
    Description: (A short description of what this script does).
%}

close all;
clear all;
addpath export_fig
wi_dataset = 'face';
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(wi_dataset);

numGroups = length(countcats(test_labels));
feature_idx = 1:size(train_featureVector,2);

train_featureVector = train_featureVector(:,feature_idx);
test_featureVector = test_featureVector(:,feature_idx);


%% calculate W for LS

W_star = train_LDA(train_featureVector, train_labels, numGroups);

test_pred = predict_LDA(W_star, test_featureVector);
test_ConfMat = confusionmat(test_labels,categorical(test_pred))
test_LDA_ClassMat = test_ConfMat./(meshgrid(countcats(test_labels))')
% mean group accuracy and std
test_acc = mean(diag(test_LDA_ClassMat))
test_std = std(diag(test_LDA_ClassMat))

figure(1);
cm = confusionchart(test_ConfMat);
st = " Testing Conf Matrix ("+wi_dataset+"), Acu = " + string(test_acc*100) + "%, SD = " + string(test_std);
cm.Title =  st;
export_fig Face_LDA_2class_test -png -transparent

%% calculate W for Fisher

W_star = train_Fisher(train_featureVector, train_labels, numGroups);
train_pred_label = train_featureVector*W_star;
test_pred_label = test_featureVector*W_star;


test_pred = predict_Fisher(train_pred_label, train_labels, test_pred_label, 8);

% Create confusion matrix
test_ConfMat = confusionmat(test_labels,categorical(test_pred))
test_fish_ClassMat = test_ConfMat./(meshgrid(countcats(test_labels))')
% mean group accuracy and std
test_acc = mean(diag(test_fish_ClassMat))
test_std = std(diag(test_fish_ClassMat))

figure(2);
cm = confusionchart(test_ConfMat);
st = " Testing Conf Matrix ("+wi_dataset+"), Acu = " + string(test_acc*100) + "%, SD = " + string(test_std);
cm.Title =  st;
export_fig Face_Fish_2class_test -png -transparent

