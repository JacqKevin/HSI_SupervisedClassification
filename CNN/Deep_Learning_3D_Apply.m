function ClassifMap =Deep_Learning_3D_Apply(M,RGB,net)
% Function to create a classification model with a deep learning structure.

% Parameters
ptch=net.Layers(1, 1).InputSize(1,1:2); % size patch

% Prediction
% ax=1:size(M,1)-ptch(1);
% ay=1:size(M,2)-ptch(2);
% idxpred=[repmat(ax,1,length(ay))' reshape(repmat(ay,length(ax),1),[],1)];

% Ypred=zeros(size(idxpred,1),1);
% h=waitbar(0);
% for i=1:size(idxpred,1)
%     waitbar(i/size(idxpred,1))
%     XX{i,1}(:,:,:)=M(idxpred(i,1):idxpred(i,1)+(ptch-1),idxpred(i,2):idxpred(i,2)+(ptch-1),:);
% end
% close(h)

[rows, columns, ~] = size(M);

blockSizeR = ptch(1); % Rows in block.
blockSizeC = ptch(2); % Columns in block.
% wholeBlockRows = rows-ptch(1);
% wholeBlockCols = columns-ptch(2);
% Now scan though, getting each block and putting it as a slice of a 3D array.
sliceNumber = 1;
h=waitbar(0);
for col = 1 : 1 : columns-ptch(2)
  for row = 1 : 1 : rows-ptch(1)
    % Let's be a little explicit here in our variables
    % to make it easier to see what's going on.
    row1 = row;
    row2 = row1 + blockSizeR - 1;
    col1 = col;
    col2 = col1 + blockSizeC - 1;
    % Extract out the block into a single subimage.
    oneBlock = M(row1:row2, col1:col2,:);
    % Assign this slice to the image we just extracted.
    XX{sliceNumber,1}(1:ptch(1), 1:ptch(2), 1:size(M,3)) = oneBlock;
    sliceNumber = sliceNumber + 1;
    waitbar(sliceNumber/((rows-ptch(1))*(columns-ptch(2))))
  end
end
close(h)

Ypred=classify(net,table(XX));

% idxy=repmat(1:size(M,2)-ptch,1,size(M,1)-ptch);
% idxx=reshape(repmat((1:size(M,1)-ptch),size(M,2)-ptch,1),[],1);

ClassifMap=reshape(Ypred,size(M,1)-ptch(1),size(M,2)-ptch(2));

% for i=1:length(idxx)
%     ClassifMap(idxx(i),idxy(i))=Ypred(i);
% end

% Display
figure;
ha(1)=subplot(211);
imagesc(double(ClassifMap))
ha(2)=subplot(212);
imagesc(RGB(ceil(ptch(1)/2):end-floor(ptch(1)/2),ceil(ptch(2)/2):end-floor(ptch(2)/2),:)*(0.5/mean(RGB(:))))
linkaxes(ha,'xy')
end