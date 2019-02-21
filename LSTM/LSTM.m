% This script defines the LSTM architecture
% prepare the 

% XTrain and Ytraind are obtained in CreateDictionary.m
% They can be loaded from XYtrain.mat
%load("XYtrain.mat");

load('XYtrain.mat');

inputSize = size(XTrain{1},1);
numHiddenUnits = 200;
numClasses = numel(categories([YTrain{:}]));

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