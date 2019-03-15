% this script uses the GN features as input of LSTM
% and the captions are used as target

addpath('../../Functions');
load('../vocabulary.mat');
clear captionsString;
clear captionsToken;
clear bow;
DS=readCaptions();
endOfTextCharacter = compose("\x2403");
load('../../ScenarioAnalysis/GNfeatures.mat')
splitCTR=[0 0 0]

% count elements for each split
for i=1:54605
    [iID,sID]=captionID2imageID(i);
    split=DS.images(iID).split;
    switch split
        case 'train'
            splitCTR=splitCTR+[1 0 0];
        case 'test'
            splitCTR=splitCTR+[0 1 0];
        case 'val'
            splitCTR=splitCTR+[0 0 1];
    end
%    fprintf("Processing Caption %d of 54605\n",i);
end
YTrain=cell(1,splitCTR(1));
%YTest=cell(1,splitCTR(2));
%YVal=cell(1,splitCTR(3));

trainYObj=matfile('YTrain.mat','Writable',true);
testYObj=matfile('YTest.mat','Writable',true);
valYObj=matfile('YTval.mat','Writable',true);

YTrainBatch={};
YTestBatch={};
YValBatch={};

batchSize=20000;

ltrain=1;ltest=1;lval=1;
trainCTR=0; testCTR=0; valCTR=0;

% create the target caption vectors
for i=1:54605%54605
    [iID,sID]=captionID2imageID(i);
    words=captionsCell{i};
    wordsShifted=[cellstr(words(1:end)')' endOfTextCharacter];
    Y = categorical(wordsShifted);
    split=DS.images(iID).split;
    if strcmp(split,'train')
        trainCTR=trainCTR+1;
        % append to the train batch vector
        YTrainBatch{1,size(YTrainBatch,2)+1}=Y;
    elseif strcmp(split,'test')
        testCTR=testCTR+1;
        YTestBatch{1,size(YTestBatch,2)+1}=Y;
    elseif strcmp(split,'val')
        valCTR=valCTR+1;
        YTValBatch{1,size(YValBatch,2)+1}=Y;
    end
    % post status update
    if mod(i,1000)==0
        p=100*(i/54605);
        fprintf("%d train\t%d test\t%d val\t %.4f%%\n",trainCTR,testCTR,valCTR,p);    % save batches of results after every 815 images
    end
    if mod(i,batchSize)==0 || i==54605
        trainYObj.YTrain(1,ltrain:trainCTR)=YTrainBatch;
        YTrainBatch={};
        ltrain=trainCTR+1;
        if ~isempty(YTestBatch)
            testYObj.YTest(1,ltest:testCTR)=YTestBatch;
            YTestBatch={};
            ltest=testCTR+1;
        end
        if ~isempty(YValBatch)
            valYObj.YVal(1,lval:valCTR)=YValBatch;
            YValBatch={};
            lval=valCTR+1;
        end
        
        p=100*(i/54605);
        fprintf("%d train\t%d test\t%d val\t %.4f%%\n",trainCTR,testCTR,valCTR,p);
        
    end
end

trainCTR=0; testCTR=0; valCTR=0;
XTrain=cell(1,splitCTR(1));
XTest=cell(1,splitCTR(2));
XVal=cell(1,splitCTR(2));
for i=1:10921
    for j=1:5
        split=DS.images(i).split;
        switch split
            case 'train'
                trainCTR=trainCTR+1;
                XTrain(1,trainCTR)={featureSet(:,i)};
            case 'test'
                testCTR=testCTR+1;
                XTest(1,testCTR)={featureSet(:,i)};
            case 'val'
                valCTR=valCTR+1;
                XVal(1,valCTR)={featureSet(:,i)};
        end
    end
    if mod(i,500)==0
        p=100*i/10921;
        fprintf("%.4f\n",p);
    end
end
        
