function w_star = train_Fisher(X, Y, classes) 


temp_Y = double(categorical(Y));

% Since we have more then on classes, simple T wont learn each class
% we need to create N*(Number of classes) label where value will be 1 if
% data belong that class

ST=zeros(size(X,2)); 
m=sum(X)/length(X); % 1x13
ST = ((X-m)'*(X-m));

SW=zeros(size(X,2));
for i=1:classes
    X_c=X(find(temp_Y(:,1)==i),:);
    mean_c=sum(X_c)'/length(X_c);
    X_c = X_c';
    SW=SW+(X_c-mean_c)*(X_c-mean_c)';
end
%size(SW)
SB=ST-SW;

[Vec,Dig]=eig(SW\SB);
[~,order] = sort(diag(Dig),'descend'); % sort
% choosing 1 dimenstion lesser the total class as mentioned in pizza
w_star=Vec(:,order(1:classes-1));

end