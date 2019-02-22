% This script emulates CreateDictionary, but with a word-by-word approach
%% Load dependencies and caption dataset
addpath('../Functions');
DS=readCaptions();
%% Store all the captions in a vector
%captions=strings(10921*5,1);
captionsCell=cell(54605,1);
captionsString=strings(54605,1);
%whitespaceCharacter=compose("\x00B7");
punctuation=["!","#","+","'",";","(",")",","];
%numbers=["0","1","2","3","4","5","6","7","8","9"];
%newlineCharacter = compose("\x00B6");

ctr=0;
for i=1:10921
    for j=1:5
        ctr=ctr+1;
        
        %captions(ctr,1)=DS.images(i).sentences(j).raw;
        cap=DS.images(i).sentences(j).raw;
        % FIX the missing data and corrupted data
        % if caption is empty, take the next one
        % remove the empty caption '1 .'
        
        if cap==" ." || strcmp(cap,'1 .')
            if j~=5
                cap=DS.images(i).sentences(j+1).raw;
            else
                cap=DS.images(i).sentences(j-1).raw;
            end
        end
        % remove punctuation and replace spaces and newline characters
        cap=replace(cap," .","");
        %cap=replace(cap," ",whitespaceCharacter);
        cap=replace(cap,punctuation,"");
        captionsCell{ctr}=cap;
        captionsString(ctr)=cap;
    end
end
%% tokenize and create bag of words
%for each caption, add a space before and after
load("wordsToReplace.mat")
captionsString(:)=strcat(" ",captionsString(:)," ");
% each word to substitute in w cannot be a subword
% i.e. it must have a space before and after
% if you want to delete a word, replace with single space
for wi=1:size(W,1)
    for wj=1:2
        if strcmp(W(wi,wj),"")
            fprintf("%d %d \n",wi,wj);
            W(wi,wj)=" ";
        elseif strcmp(W(wi,wj),"`") || strcmp(W(wi,wj),"]")
            % special characters
            continue
        else
            W(wi,wj)=strcat(" ",W(wi,wj)," ");
        end
    end
end
captionsString=replaceWords(W,captionsString);
% return situation with spaces before and after string back to normal
strip(captionsString(:));
captionsToken=tokenizedDocument(captionsString);
bow=bagOfWords(captionsToken);
vocabulary=(sort(bow.Vocabulary))';