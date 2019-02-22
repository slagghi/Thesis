% This script generates a random character sequence
% run after CreateDictionary.m
initialCharacters=extractBefore(textData,2);
firstCharacter=datasample(initialCharacters,1);
generatedText=string(firstCharacter);
X=zeros(numUniqueCharacters,1);
idx=strfind(uniqueCharacters,firstCharacter);
X(idx)=1;
vocabulary=string(net.Layers(end).Classes);

maxLength = 500;
while strlength(generatedText) < maxLength
    % Predict the next character scores.
    [net,characterScores] = predictAndUpdateState(net,X,'ExecutionEnvironment','cpu');
    
    % Sample the next character.
    newCharacter = datasample(vocabulary,1,'Weights',characterScores);
    
    % Stop predicting at the end of text.
    if newCharacter == endOfTextCharacter
        break
    end
    
    % Add the character to the generated text.
    generatedText = generatedText + newCharacter;
    
    % Create a new vector for the next input.
    X(:) = 0;
    idx = strfind(uniqueCharacters,newCharacter);
    X(idx) = 1;
end
generatedText = replace(generatedText,[newlineCharacter whitespaceCharacter],[newline " "])
