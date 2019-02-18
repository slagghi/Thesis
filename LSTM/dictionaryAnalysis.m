% this performs an analysis of the vocabulry of the captions
addpath('../Functions')
dataset=readCaptions;
list={};
ctr=0;
punctuationCharacters = [" " "'" "(" "." "," "’" ")" ":" "?" "!"];
%% Results for this chunk are saved in wordList.mat
disp("---------- BUILDING DICTIONARY ----------");
for i=1:10921
    for j=1:5
        caption=dataset.images(i).sentences(j).tokens;
        for k=1:size(caption,1)
            % remove punctuation characters
            candidate=caption{k};
            candidate=replace(candidate,punctuationCharacters,"");
            if(candidate=="")
                continue
            end
            if ismember(candidate,list)
                continue
            else
                ctr=ctr+1;
                list{ctr}=candidate;
            end
        end
    end
    % checkpoint progression output
    if mod(i,500)==0
        progress=i/10921;
        fprintf('Parsed %d images\t progress:%.3f%%\n',i,progress*100);
    end
end
%sort the word list in alphabetical order
list=sort(list);

%load('wordList.mat')

%punctuationCharacters = ["." "," "’" ")" ":" "?" "!"];
%wordList{:} = replace(wordList{:}," " + punctuationCharacters,punctuationCharacters);

%% count word occurrences
dict={};
disp('---------- COUNTING WORD OCCURRENCES ----------');
for w=1:size(list,2)
    word=list{w};
    fprintf('\nNow analysing: %s\t Occurrences found: ',list{w});
    occurrences=0;
    for i=1:10921
        for j=1:5
            caption=dataset.images(i).sentences(j).tokens;
            for k=1:size(caption,1)
                if strcmp(caption{k},word)
                    occurrences=occurrences+1;
                    % prints progress
                    digits=numel(num2str(occurrences-1));
                    for d=1:digits
                        fprintf('\b')
                    end
                    fprintf('%d',occurrences)
                end
            end
        end
    end
    dict{w,1}=list{w};
    dict{w,2}=occurrences;
    progress=w/size(list,2)*100;
    if mod(w,10)==0
        fprintf('\n---------- Progress: %.4f%% ----------',progress)
    end
end
fprintf('\n')