%% Create input data
% filename = "sonnets.txt";
% textData = fileread(filename);
% 
% textData = replace(textData,"  ","");
% textData = split(textData,[newline newline]);
% textData = textData(5:2:end);
% 
% newlineCharacter = compose("\x00B6");
% whitespaceCharacter = compose("\x00B7");
% textData = replace(textData,[newline " "],[newlineCharacter whitespaceCharacter]);
% 
% % test: replace with captions
% textData=captions(7111);
% 
% uniqueCharacters = unique([textData{:}]);
% numUniqueCharacters = numel(uniqueCharacters);
% 
% endOfTextCharacter = compose("\x2403");
% for i = 1:numel(textData)
%     characters = textData{i};
%     sequenceLength = numel(characters);
%     
%     % Get indices of characters.
%     [~,idx] = ismember(characters,uniqueCharacters);
%     
%     % Convert characters to vectors.
%     X = zeros(numUniqueCharacters,sequenceLength);
%     for j = 1:sequenceLength
%         X(idx(j),j) = 1;
%     end
%     
%     % Create vector of categorical responses with end of text character.
%     charactersShifted = [cellstr(characters(2:end)')' endOfTextCharacter];
%     Y = categorical(charactersShifted);
%     
%     XTrain{i} = X;
%     YTrain{i} = Y;
% end
% 
% foo=1;


load('XYtrain.mat');
%XTrain=XTrain(10306);
%YTrain=YTrain(10306);

%% Network architecture definition
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

%% Train the network
net = trainNetwork(XTrain,YTrain,layers,options);