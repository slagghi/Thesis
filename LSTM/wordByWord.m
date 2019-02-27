% This script emulates CreateDictionary, but with a word-by-word approach
%% Load dependencies and caption dataset
addpath('../Functions');
DS=readCaptions();
%% Store all the captions in a vector
%captions=strings(10921*5,1);
%captionsCell=cell(54605,1);
%captionsString=strings(54605,1);
%whitespaceCharacter=compose("\x00B7");
%punctuation=["!","#","+","'",";","(",")",","];
%numbers=["0","1","2","3","4","5","6","7","8","9"];
%newlineCharacter = compose("\x00B6");

% ctr=0;
% for i=1:10921
%     for j=1:5
%         ctr=ctr+1;
%         
%         %captions(ctr,1)=DS.images(i).sentences(j).raw;
%         cap=DS.images(i).sentences(j).raw;
%         % FIX the missing data and corrupted data
%         % if caption is empty, take the next one
%         % remove the empty caption '1 .'
%         
%         if cap==" ." || strcmp(cap,'1 .')
%             if j~=5
%                 cap=DS.images(i).sentences(j+1).raw;
%             else
%                 cap=DS.images(i).sentences(j-1).raw;
%             end
%         end
%         % remove punctuation and replace spaces and newline characters
%         cap=replace(cap," .","");
%         %cap=replace(cap," ",whitespaceCharacter);
%         cap=replace(cap,punctuation,"");
%         captionsCell{ctr}=cap;
%         captionsString(ctr)=cap;
%     end
% end
% %% tokenize and create bag of words
% %for each caption, add a space before and after
% load("wordsToReplace.mat")
% captionsString(:)=strcat(" ",captionsString(:)," ");
% % each word to substitute in w cannot be a subword
% % i.e. it must have a space before and after
% % if you want to delete a word, replace with single space
% for wi=1:size(W,1)
%     for wj=1:2
%         if strcmp(W(wi,wj),"")
%             fprintf("%d %d \n",wi,wj);
%             W(wi,wj)=" ";
%         elseif strcmp(W(wi,wj),"`") || strcmp(W(wi,wj),"]")
%             % special characters
%             continue
%         else
%             W(wi,wj)=strcat(" ",W(wi,wj)," ");
%         end
%     end
% end
% captionsString=replaceWords(W,captionsString);
% % return situation with spaces before and after string back to normal
% strip(captionsString(:));
% captionsToken=tokenizedDocument(captionsString);
% bow=bagOfWords(captionsToken);
% vocabulary=(sort(bow.Vocabulary))';
%% Create test, train and val vectors
% Results of previous code are saved in vocabulary.mat
load('vocabulary.mat');
clear captionsString;
clear captionsToken;
clear bow;
csvfile='datastore.csv';
%XTrain=cell(1,10);
%YTrain=cell(1,10);
XVal={};YVal={};
endOfTextCharacter = compose("\x2403");
DS=readCaptions();

%XTrain=cell(1,21835);
%YTrain=cell(1,43670);
trainCTR=0;
valCTR=0;
% SPLIT: xtrain1 for i=1:27535
%        xtrain2 for i=27535:54605...
iterStart=1;
iterEnd=56405;
for i=iterStart:iterEnd%54605
    % divide in train and test splits
    [imageID,sentenceID]=captionID2imageID(i);
    if imageID>10921
        continue
    end
    split=DS.images(imageID).split;
   if strcmp(split,'train')
       trainCTR=trainCTR+1;
       words=captionsCell{i};
%       sentenceVector=vectorEncode(words,vocabulary);
%       XTrain{trainCTR}=sentenceVector;
       wordsShifted=[cellstr(words(2:end)')' endOfTextCharacter];
       Y = categorical(wordsShifted);
       YTrain{trainCTR}=Y;
  end
%     if strcmp(split,'val')
%         valCTR=valCTR+1;
%         words=captionsCell{i};
%         sentenceVector=vectorEncode(words,vocabulary);
%         XVal{valCTR}=sentenceVector;
%         wordsShifted=[cellstr(words(2:end)')' endOfTextCharacter];
%         Y=categorical(wordsShifted);
%         YVal{valCTR}=Y;
%     end
    if mod(i,100)==0
        p=100*(i/iterEnd);
        fprintf('processing caption %d of %d (%.4f%%)\n',i,iterEnd,p);
    end
 end

%% Network architecture definition
inputSize = size(XTrain{1},1);
numHiddenUnits = 400;
numClasses = numel(categories([YTrain{1:10000}]));



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

% Train the network
net = trainNetwork(XTrain(1:10000),YTrain(1:10000),layers,options);
