% This script generates the dictionary by reading all the captions in the
% dataset and tokenizing them

%% Load dependencies and caption dataset
addpath('../Functions');
DS=readCaptions();
%% Store all the captions in a vector
%captions=strings(10921*5,1);
captions=cell(54605,1);

whitespaceCharacter=compose("\x00B7");
punctuation=["!","#","+","'",";","(",")",","];
numbers=["0","1","2","3","4","5","6","7","8","9"];
newlineCharacter = compose("\x00B6");

ctr=0;
for i=1:10921
    for j=1:5
        ctr=ctr+1;

        % DEBUG stop
        if ctr==10310
            fprintf("DEBUG START\n");
        end
        
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
        cap=replace(cap," ",whitespaceCharacter);
        cap=replace(cap,punctuation,"");
        captions{ctr}=cap;
    end
end
% remove unwanted punctuation
%captions(:)=replace(captions(ctr,1),"  "," ");


uniqueCharacters = unique([captions{:}]);
numUniqueCharacters = numel(uniqueCharacters);

%% Encode captions as a matrix
% each character is a one-hot column vector
endOfTextCharacter = compose("\x2403");
for i = 1:numel(captions)
    characters = captions{i};
    sequenceLength = numel(characters);
    
    % Get indices of characters.
    [~,idx] = ismember(characters,uniqueCharacters);
    
    % Convert characters to vectors.
    X = zeros(numUniqueCharacters,sequenceLength);
    for j = 1:sequenceLength
        X(idx(j),j) = 1;
    end
    
    % Create vector of categorical responses with end of text character.
    charactersShifted = [cellstr(characters(2:end)')' endOfTextCharacter];
    Y = categorical(charactersShifted);
    
    XTrain{i} = X;
    YTrain{i} = Y;
end

%% Tokenize
%documents=tokenizedDocument(captions);
%ds = documentGenerationDatastore(documents);
%ds = sort(ds);