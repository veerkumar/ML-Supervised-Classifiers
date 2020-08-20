function w_star = train_LDA(X, Y, classes) 

X = [ones(size(X,1),1) X]; % N*(No_feature+1)
T = zeros(length(Y),classes); % N*No_classes 
temp_Y = double(categorical(Y));

% Since we have more then on classes, simple T wont learn each class
% we need to create N*(Number of classes) label where value will be 1 if
% data belong that class

for i=1:length(temp_Y)
   T(i,temp_Y(i,1)) = 1;
end

%Psudu inverst of X * true label
w_star =  pinv(X)* T; % no_class*(no_Feature+1)

end