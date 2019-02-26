function oH = oneHot(n,dictSize)
% This function receives an integer as input and returns the corresponding
% one-hot vector as output

% if no dictionary is passed, use the size of the vocabulary
 if ~exist('dictSize','var')
     % third parameter does not exist, so default it to something
      dictSize = 2595;
 end
oH=zeros(dictSize,1);
oH(n)=1;
end

