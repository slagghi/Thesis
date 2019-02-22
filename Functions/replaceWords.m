function [captionsString] = replaceWords(wordMatrix,captionsString)
% This functions replaces a list of word in broken english in the caption
% dataset
for i=1:size(wordMatrix,1)
    word1=wordMatrix(i,1);
    word2=wordMatrix(i,2);
    fprintf("Replacing %s with %s\n",word1,word2);
    captionsString(:)=replace(captionsString(:),word1,word2);
end

