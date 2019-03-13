function [imageID,sentenceID] = captionID2imageID(captionID)
% This function takes as input a caption ID, and returns the ID of the
% image it was taken from, as well as the caption# (1~5) for that image
imageID=floor(captionID/5)+1;
sentenceID=mod(captionID,5)+1;

% Verbose mode: print the inputs and output
% fprintf("caption %d refers to image %d, sentence %d\n",captionID,imageID,sentenceID);
end

