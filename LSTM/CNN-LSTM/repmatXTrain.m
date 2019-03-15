% this code changes XTrain to have the same length of the expected output
% sentence

load('YTrain_firstCaption.mat')
load('XTrain.mat')


for i=1:8734
    r=size(YTrain_firstCaption{i},2);
    rep=repmat(XTrain{i},1,r);
    XTrain{i}=rep;
end