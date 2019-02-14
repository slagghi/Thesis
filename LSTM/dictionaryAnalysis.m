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
for w=1:12
    word=bitelist{w};
    occurrences=0;
    for i=1:10921
        for j=1:5
            caption=dataset.images(i).sentences(j).tokens;
            for k=1:size(caption,1)
                if strcmp(caption{k},word)
                    occurrences=occurrences+1;
                end
            end
        end
        dict{w,1}=bitelist{w};
        dict{w,2}=occurrences;
    end
end