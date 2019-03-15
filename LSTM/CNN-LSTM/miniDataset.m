% This script creates a minidataset for experimenting with malab LSTMs
load('XTrain.mat')
addpath('../../Functions')
DS=readCaptions();
trainSentences={};
ctr=0;
for i=1:1000
    if strcmp(DS.images(i).split,'train')
        ctr=ctr+1;
        trainSentences{ctr}=DS.images(i).sentences(1).tokens;
    end
end
XTrain=XTrain(1,1:798);