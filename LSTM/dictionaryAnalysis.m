% this performs an analysis of the vocabulry of the captions
addpath('../Functions')
dataset=readCaptions;
list={};
ctr=0;

%% Results for this chunk are saved in wordList.mat
% for i=1:10921
%     for j=1:5
%         caption=dataset.images(i).sentences(j).tokens;
%         for k=1:size(caption,1)
%             if ismember(caption{k},list)
%                 continue
%             else
%                 ctr=ctr+1;
%                 list{ctr}=caption{k};
%             end
%         end
%     end
%     % checkpoint progression output
%     if mod(i,100)==0
%         fprintf('Parsed %d images\n',i);
%     end
% end
% sort the word list in alphabetical order
% sort(list);
load('wordList.mat')
%% count word occurrences
dict={};
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
        end12
        dict{w,1}=list{w};
        dict{w,2}=occurrences;
        progress=w/size(list,2);
        if mod(progress,0.5)==0
            fprintf('\n----------------Progress: %d\%\n---------',progress)
        end
    end
end
fprintf('\n')