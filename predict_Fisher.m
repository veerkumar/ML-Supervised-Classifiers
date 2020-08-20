% This function uses KNN classifier to assign the classes to predicted
% label

function [predicted_label] = predict_Fisher(train_data, train_label, test_data, K) 

%find distance of each test data from other 
train_label = double(train_label);
predicted_label = zeros(length(test_data),1);

for i=1:size(test_data,1)
    temp = repmat(test_data(i,:)',1,length(train_data));
    dist =  sum((temp -train_data').^2,1);
    [~, order] = sort(dist);
    %rearange train_label accordingt to the distance and then select K
    %labels
    temp_label = train_label(order');
    temp_label = temp_label(1:K);
    predicted_label(i,1) = mode(temp_label);
end

end