function sentenceVector = vectorEncode(sentence,vocabulary)
% This function takes a sentence (string array) as input and encodes it as a sequence of
% word vectors (one-hot encoding)
l=length(sentence);
dictSize=size(vocabulary,1);
sentenceVector=zeros(dictSize,l);
for i=1:l
    word=sentence(i);
    word1h=word21h(word,vocabulary);
    sentenceVector(:,i)=word1h;
end
end

