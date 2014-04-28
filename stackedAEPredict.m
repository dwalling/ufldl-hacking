function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

activity = cell(numel(stack)+1, 1);
% Calculate feed-forward pass through stack
activity{1}.a = data;
for i=2:numel(activity)
    activity{i}.z = bsxfun(@plus, stack{i-1}.w * activity{i-1}.a, stack{i-1}.b);
    activity{i}.a = sigmoid(activity{i}.z);
end

% Forward through softmax

td = softmaxTheta * activity{numel(activity)}.a;
ethx = exp(bsxfun(@minus, td, max(td, [], 1)));
p_j_given_x = bsxfun(@rdivide, ethx, sum(ethx));
[p_pred, pred] = max(p_j_given_x);
% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
