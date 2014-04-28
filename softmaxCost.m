function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

td = theta*data;
ethx = exp(bsxfun(@minus, td, max(td, [], 1)));
p_j_given_x = bsxfun(@rdivide, ethx, sum(ethx));
cost = -mean(sum(groundTruth .* log(p_j_given_x), 1));
cost = cost + 0.5 * lambda * sum(sum(theta.*theta, 1), 2);

for j = 1:numClasses
    thetagrad(j,:) = -mean(bsxfun(@times, data, (groundTruth(j, :) - p_j_given_x(j, :))), 2)';
end

thetagrad = thetagrad + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

