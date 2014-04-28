function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                     
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
activity = cell(numel(stack)+1, 1);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

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

cost = -mean(sum(groundTruth .* log(p_j_given_x), 1));
cost = cost + 0.5 * lambda * sum_sq_W(softmaxTheta);

% no weight decay on stack
% no sparsity penalty

% Calculate gradients

for j = 1:numClasses
    softmaxThetaGrad(j,:) = -mean(bsxfun(@times, activity{numel(activity)}.a, ...
                                    (groundTruth(j, :) - p_j_given_x(j, :))), 2)';
end

softmaxThetaGrad = softmaxThetaGrad + lambda * softmaxTheta;

activity{numel(activity)}.delta = -softmaxTheta' * (groundTruth - p_j_given_x) .* ...
                       (activity{numel(activity)}.a .* (1 - activity{numel(activity)}.a));
                   
for i = (numel(stack)):-1:1
    activity{i}.delta = stack{i}.w' * activity{i+1}.delta .* ...
                        (activity{i}.a .* (1 - activity{i}.a));
    stackgrad{i}.w = (activity{i+1}.delta * activity{i}.a') / M;
    stackgrad{i}.b = mean(activity{i+1}.delta, 2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function ssw = sum_sq_W(W)
    ssw = sum(sum(W .* W, 1), 2);
end
