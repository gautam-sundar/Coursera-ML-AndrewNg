function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

X_label_0 = X(find(y == 0),:)
plot(X_label_0(:,1), X_label_0(:,2), 'r+')
X_label_1 = X(find(y == 1), :)
plot(X_label_1(:,1), X_label_1(:,2), 'go')







% =========================================================================



hold off;

end
