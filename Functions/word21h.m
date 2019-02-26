function oH = word21h(word,vocabulary)
%This function takes a word as input and returns its corresponding one-hot
%vector
index=find(vocabulary==word);
dictSize=size(vocabulary,1);
oH=oneHot(index,dictSize);
end

