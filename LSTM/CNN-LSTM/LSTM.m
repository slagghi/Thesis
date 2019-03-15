% this code defines the LSTM to predict the captions from the Googlenet
% features.

inputSize = 1000; % # of Googlenet features
numHiddenUnits = 1000;
numClasses = numel(categories([YTrain{:}])); % # of words in the vocabulary

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'MiniBatchSize',67,...
    'Plots','training-progress', ...
    'Verbose',false);

%% Train the network
net = trainNetwork(XTrain,YTrain,layers,options);