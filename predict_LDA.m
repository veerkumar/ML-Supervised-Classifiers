function [predicted_label] = predict_LDA(W_star, X) 

%size(W_star) = no_class*(no_Feature+1)

X = [ones(size(X,1),1) X]; % N*(No_feature+1)
predicted_label = zeros(size(X,1),1);

% W_Star is NO_CLASSES*(No_features+1),
% for each training data we will test with every class and which ever is
% maximum, classify it to that class

for i=1:length(X)
    row = X(i,:);
    class_values = W_star'*row';
    [~,index] = max(class_values); % select maximum
    predicted_label(i,1) = index;
end

end