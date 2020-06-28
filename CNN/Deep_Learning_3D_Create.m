function [net,info,Res] =Deep_Learning_3D_Create(M,RGB,Cartlab,poolnum,numN,numP)%,ptch,poolnum,nlayer)
% Function to create a classification model with a deep learning structure.

close all

if nargin<5
    list={'Layers: 4 (3-2...D)','Layers: 6 (3-2...D)',...
        'Layers: 6 (3-3-2...D)','Layers: 6 (3-2-3-2...D)',...
        'Layers: 8 (3-2...D)','Layers: 8 (3-3-2...D)',...
        'Layers: 8 (3-2-3-2...D)','Layers: 8 (3-3-3-2...D)',...
        'Layers: 8 (3-2-3-2-3-2...D)',...
        'Layers: 10 (3-2...D)','Layers: 10 (3-3-2...D)',...
        'Layers: 10 (3-2-3-2...D)'};%,'Layers: 10 (3-3-3-2...D)',...
    %'Layers: 10 (3-2-3-2-3-2...D)'};
    [indN,~] = listdlg('ListString',list);
else
    indN=numN;
end

if nargin<6
    list={'patch: 5*5','patch: 3*3'};
    [indP,~] = listdlg('ListString',list);
    numP=0;
else
    indP=0;
end

if indP==1||numP==5
    ptch=[5 5];
else if indP==2||numP==3
        ptch=[3 3];
    end
end

if iscell(M)
    nM=max(size(M));
else
    nM=1;
    Mtp=M;RGBtp=RGB;Cartlabtp=Cartlab;
    clear M RGB Cartlab
    M{1}=Mtp;
    RGB{1}=RGBtp;
    Cartlab{1}=Cartlabtp;
    clear Mtp RGBtp Cartlabtp
end

itermi=0;
numim=[];label=[];
for iM=1:nM
    % Display labelled map
    figure;
    ha(1)=subplot(211);
    imagesc(RGB{iM}*(0.5/mean(RGB{iM}(:))))
    ha(2)=subplot(212);
    imagesc(Cartlab{iM});
    linkaxes(ha,'xy')
    
    % Find the number of labeled classes
    m=max(Cartlab{iM}(:));
    
    % Find the positions of the labeled areas
    stats=[];
    c=0;
    idx=[];length_idx=[];
    for i=1:m
        stats= regionprops(Cartlab{iM}==i);
        for j=1:length(stats)
            idx(j+c,1)=i;
            idx(j+c,2:5)=[floor(stats(j).BoundingBox(1)) floor(stats(j).BoundingBox(1))+stats(j).BoundingBox(3) floor(stats(j).BoundingBox(2)) floor(stats(j).BoundingBox(2))+stats(j).BoundingBox(4)];
            
            length_idx(j+c,1:2)=[stats(j).BoundingBox(3) stats(j).BoundingBox(4)]; %dx dy
        end
        c=length(stats)+c;
    end
    
    % Subsampling of each areas
    num_sub=floor(length_idx./(ptch-1));
    
    % Subsampling of the map
    iter=1;
    areas=zeros(sum(num_sub(:)),5);
    for i=1:size(idx,1)
        for j=1:num_sub(i,1)
            for k=1:num_sub(i,2)
                areas(iter,1)=idx(i,1);
                areas(iter,2:5)=[idx(i,2)+(j-1)*(ptch(1)-1) idx(i,2)+(j)*(ptch(1)-1) idx(i,4)+(k-1)*(ptch(2)-1) idx(i,4)+(k)*(ptch(2)-1)];
                iter=iter+1;
            end
        end
    end
    
    % Subsampling of the image (5*5*channel*nb patch)
    label=[label; areas(:,1)];
    % Y=zeros(size(label,1),max(label(:)));
    % for i=1:size(Y)
    %    Y(i,label(i))=1;
    % end
    X=[];
    for i=1:size(areas,1)
        X(:,:,:,itermi+i)=M{iM}(areas(i,4):areas(i,5),areas(i,2):areas(i,3),:);
        Xtp(:,:,:,i)=M{iM}(areas(i,4):areas(i,5),areas(i,2):areas(i,3),:);
    end
    itermi=itermi+size(areas,1);
    
    %num image
    numim=[numim ones(1,size(areas,1))*iM];
    numtp=ones(1,size(areas,1))*iM;
    
    % Set definition with same size in each class
    [C,~,~] = unique(areas(:,1)); % number of class
    for i=1:length(C)
        numc(i)=sum(areas(:,1)==C(i)); % number per class
    end
    cal=round(0.7*min(numc));  % 70% in the training set
    for i=1:length(C)
        al = rand(1,numc(i)); % random number
        [~,iii] = sort(al);
        [tp,~]=find(areas(:,1)==C(i));
        if iM==1&&i==1
            Xtrain=Xtp(:,:,:,tp(iii(1:cal)));
            Xtest=Xtp(:,:,:,tp(iii(cal+1:end)));
            Ytrain=areas(tp(iii(1:cal)),1);
            Ytest=areas(tp(iii(cal+1:end)),1);
            numimtrain=numtp(tp(iii(1:cal)));
            numimtest=numtp(tp(iii(cal+1:end)));
        else
            Xtrain=cat(4,Xtrain,Xtp(:,:,:,tp(iii(1:cal))));
            Xtest=cat(4,Xtest,Xtp(:,:,:,tp(iii(cal+1:end))));
            Ytrain=[Ytrain; areas(tp(iii(1:cal)),1)];
            Ytest=[Ytest; areas(tp(iii(cal+1:end)),1)];
            numimtrain=[numimtrain numtp(tp(iii(1:cal)))];
            numimtest=[numimtest numtp(tp(iii(cal+1:end)))];
        end
    end
end
clear numtp Xtp
Ytrain=categorical(Ytrain);
Ytest=categorical(Ytest);

for i=1:size(Xtrain,4)
    Xtrainc{i,1}=squeeze(Xtrain(:,:,:,i));
end
for i=1:size(Xtest,4)
    Xtestc{i,1}=squeeze(Xtest(:,:,:,i));
end

figure;
subplot(121)
for i=1:4
    for j=1:length(C)
        atrain(i,j)=length(Ytrain(numimtrain'==i&Ytrain==categorical(C(j))));
    end
end
bar(atrain','stacked')
grid on
subplot(122)
for i=1:4
    for j=1:length(C)
        atest(i,j)=length(Ytest(numimtest'==i&Ytest==categorical(C(j))));
    end
end
bar(atest','stacked')
grid on

%% Deep Learning
% Parameters
epoch=50;
minibatch=round(epoch/600*size(Ytrain,1));
if minibatch<5
    minibatch=5;
end
if nargin<5
    poolnum=5;
end

% numberOfWorkers = 15;
% poolobj =parpool(numberOfWorkers);

options = trainingOptions('sgdm',... %
    'InitialLearnRate',0.01, ... %
    'Momentum',0.9,... %
    'L2Regularization',0.0001,...
    'MaxEpochs',epoch,... %
    'MiniBatchSize',minibatch,... %
    'LearnRateSchedule','piecewise',...
    'Shuffle','every-epoch',...
    'GradientThresholdMethod','l2norm',...
    'GradientThreshold',0.05, ...
    'Plots','training-progress', ...
    'VerboseFrequency',10,...
    'WorkerLoad',0.95,...
    'ValidationData',table(Xtestc,Ytest));
%    'ExecutionEnvironment','parallel',...

% Define Network Architecture
if indN==7
    layers = [
        image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
        
        % Conv1
        convolution3dLayer([3 3 3],20,...
        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu1
        reluLayer
        
        % ConvPool2
        convolution3dLayer([1 1 3],poolnum,...
        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu2
        reluLayer
        
        % Conv3
        convolution3dLayer([3 3 3],35,...
        'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu3
        reluLayer
        
        % ConvPool4
        convolution3dLayer([1 1 2],poolnum,...
        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu4
        reluLayer
        
        % Conv5
        convolution3dLayer([1 1 3],35,...
        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu5
        reluLayer
        
        % ConvPool6
        convolution3dLayer([1 1 1],poolnum,...
        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu6
        reluLayer
        
        % Conv7
        convolution3dLayer([1 1 3],35,...
        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu7
        reluLayer
        
        % ConvPool8
        convolution3dLayer([1 1 1],poolnum*2,...
        'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu8
        reluLayer
        
        fullyConnectedLayer(2,...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        softmaxLayer
        classificationLayer];
else if indN==9
        layers = [
            image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
            
            % Conv1
            convolution3dLayer([3 3 3],20,...
            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu1
            reluLayer
            
            % ConvPool2
            convolution3dLayer([1 1 3],poolnum,...
            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu2
            reluLayer
            
            % Conv3
            convolution3dLayer([3 3 3],35,...
            'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu3
            reluLayer
            
            % ConvPool4
            convolution3dLayer([1 1 3],poolnum,...
            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu4
            reluLayer
            
            % Conv5
            convolution3dLayer([3 3 3],35,...
            'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu5
            reluLayer
            
            % ConvPool6
            convolution3dLayer([1 1 2],poolnum,...
            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu6
            reluLayer
            
            % Conv7
            convolution3dLayer([1 1 2],35,...
            'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu7
            reluLayer
            
            % ConvPool8
            convolution3dLayer([1 1 3],poolnum*2,...
            'Stride',[2 2 2],'Padding',[0 0 1; 0 0 1],...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            
            % Relu8
            reluLayer
            
            fullyConnectedLayer(2,...
            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
            'BiasLearnRateFactor',2,'BiasL2Factor',0)
            softmaxLayer
            classificationLayer];
    else if indN==3
            layers = [
                image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                
                % Conv1
                convolution3dLayer([3 3 3],20,...
                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                
                % Relu1
                reluLayer
                
                % ConvPool2
                convolution3dLayer([3 3 3],poolnum,...
                'Stride',[2 2 2],'Padding',[1 1 1; 1 1 1],...
                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                
                % Relu2
                reluLayer
                
                % Conv3
                convolution3dLayer([1 1 3],35,...
                'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                
                % Relu3
                reluLayer
                
                % ConvPool4
                convolution3dLayer([1 1 2],poolnum,...
                'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                
                % Relu4
                reluLayer
                
                % Conv5
                convolution3dLayer([1 1 3],35,...
                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                
                % Relu5
                reluLayer
                
                % ConvPool6
                convolution3dLayer([1 1 3],poolnum,...
                'Stride',[2 2 2],'Padding',[0 0 1; 0 0 1],...
                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                
                % Relu6
                reluLayer
                
                fullyConnectedLayer(2,...
                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                softmaxLayer
                classificationLayer];
        else if indN==2
                layers = [
                    image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                    
                    % Conv1
                    convolution3dLayer([3 3 3],20,...
                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                    
                    % Relu1
                    reluLayer
                    
                    % ConvPool2
                    convolution3dLayer([1 1 3],poolnum,...
                    'Stride',[2 2 2],'Padding',[1 1 1; 1 1 1],...
                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                    
                    % Relu2
                    reluLayer
                    
                    % Conv3
                    convolution3dLayer([1 1 3],35,...
                    'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                    
                    % Relu3
                    reluLayer
                    
                    % ConvPool4
                    convolution3dLayer([1 1 2],poolnum,...
                    'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                    
                    % Relu4
                    reluLayer
                    
                    % Conv5
                    convolution3dLayer([1 1 3],35,...
                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                    
                    % Relu5
                    reluLayer
                    
                    % ConvPool6
                    convolution3dLayer([1 1 3],poolnum,...
                    'Stride',[2 2 2],'Padding',[0 0 1; 0 0 1],...
                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                    
                    % Relu6
                    reluLayer
                    
                    fullyConnectedLayer(2,...
                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                    softmaxLayer
                    classificationLayer];
            else if indN==1
                    layers = [
                        image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                        
                        % Conv1
                        convolution3dLayer([3 3 3],20,...
                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                        
                        % Relu1
                        reluLayer
                        
                        % ConvPool2
                        convolution3dLayer([1 1 3],poolnum,...
                        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                        
                        % Relu2
                        reluLayer
                        
                        % Conv3
                        convolution3dLayer([1 1 3],35,...
                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                        
                        % Relu3
                        reluLayer
                        
                        % ConvPool4
                        convolution3dLayer([1 1 2],poolnum,...
                        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                        
                        % Relu4
                        reluLayer
                        
                        fullyConnectedLayer(2,...
                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                        softmaxLayer
                        classificationLayer];
                else if indN==12
                        layers = [
                            image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                            
                            % Conv1
                            convolution3dLayer([3 3 3],20,...
                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu1
                            reluLayer
                            
                            % ConvPool2
                            convolution3dLayer([1 1 3],poolnum,...
                            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu2
                            reluLayer
                            
                            % Conv3
                            convolution3dLayer([3 3 3],35,...
                            'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu3
                            reluLayer
                            
                            % ConvPool4
                            convolution3dLayer([1 1 2],poolnum,...
                            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu4
                            reluLayer
                            
                            % Conv5
                            convolution3dLayer([1 1 3],35,...
                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu5
                            reluLayer
                            
                            % ConvPool6
                            convolution3dLayer([1 1 1],poolnum,...
                            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu6
                            reluLayer
                            
                            % Conv7
                            convolution3dLayer([1 1 3],35,...
                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu7
                            reluLayer
                            
                            % ConvPool8
                            convolution3dLayer([1 1 1],poolnum*2,...
                            'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu8
                            reluLayer
                            
                            % Conv9
                            convolution3dLayer([1 1 3],35,...
                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu9
                            reluLayer
                            
                            % ConvPool10
                            convolution3dLayer([1 1 1],poolnum*2,...
                            'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            
                            % Relu10
                            reluLayer
                            
                            fullyConnectedLayer(2,...
                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                            softmaxLayer
                            classificationLayer];
                    else if indN==4
                            layers = [
                                image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                
                                % Conv1
                                convolution3dLayer([3 3 3],20,...
                                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                
                                % Relu1
                                reluLayer
                                
                                % ConvPool2
                                convolution3dLayer([1 1 3],poolnum,...
                                'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                
                                % Relu2
                                reluLayer
                                
                                % Conv3
                                convolution3dLayer([3 3 3],35,...
                                'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                
                                % Relu3
                                reluLayer
                                
                                % ConvPool4
                                convolution3dLayer([1 1 2],poolnum,...
                                'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                
                                % Relu4
                                reluLayer
                                
                                % Conv5
                                convolution3dLayer([1 1 3],35,...
                                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                
                                % Relu5
                                reluLayer
                                
                                % ConvPool6
                                convolution3dLayer([1 1 1],poolnum,...
                                'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                
                                % Relu6
                                reluLayer
                                
                                fullyConnectedLayer(2,...
                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                softmaxLayer
                                classificationLayer];
                        else if indN==5
                                layers = [
                                    image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                    
                                    % Conv1
                                    convolution3dLayer([3 3 3],20,...
                                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu1
                                    reluLayer
                                    
                                    % ConvPool2
                                    convolution3dLayer([1 1 3],poolnum,...
                                    'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu2
                                    reluLayer
                                    
                                    % Conv3
                                    convolution3dLayer([1 1 3],35,...
                                    'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu3
                                    reluLayer
                                    
                                    % ConvPool4
                                    convolution3dLayer([1 1 2],poolnum,...
                                    'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu4
                                    reluLayer
                                    
                                    % Conv5
                                    convolution3dLayer([1 1 3],35,...
                                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu5
                                    reluLayer
                                    
                                    % ConvPool6
                                    convolution3dLayer([1 1 1],poolnum,...
                                    'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu6
                                    reluLayer
                                    
                                    % Conv7
                                    convolution3dLayer([1 1 3],35,...
                                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu7
                                    reluLayer
                                    
                                    % ConvPool8
                                    convolution3dLayer([1 1 1],poolnum*2,...
                                    'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    
                                    % Relu8
                                    reluLayer
                                    
                                    fullyConnectedLayer(2,...
                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                    softmaxLayer
                                    classificationLayer];
                            else if indN==6
                                    layers = [
                                        image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                        
                                        % Conv1
                                        convolution3dLayer([3 3 3],20,...
                                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu1
                                        reluLayer
                                        
                                        % ConvPool2
                                        convolution3dLayer([3 3 3],poolnum,...
                                        'Stride',[2 2 2],'Padding',[1 1 1; 1 1 1],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu2
                                        reluLayer
                                        
                                        % Conv3
                                        convolution3dLayer([1 1 3],35,...
                                        'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu3
                                        reluLayer
                                        
                                        % ConvPool4
                                        convolution3dLayer([1 1 2],poolnum,...
                                        'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu4
                                        reluLayer
                                        
                                        % Conv5
                                        convolution3dLayer([1 1 3],35,...
                                        'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu5
                                        reluLayer
                                        
                                        % ConvPool6
                                        convolution3dLayer([1 1 2],poolnum,...
                                        'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu6
                                        reluLayer
                                        
                                        % Conv7
                                        convolution3dLayer([1 1 3],35,...
                                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu7
                                        reluLayer
                                        
                                        % ConvPool8
                                        convolution3dLayer([1 1 3],poolnum,...
                                        'Stride',[2 2 2],'Padding',[0 0 1; 0 0 1],...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        
                                        % Relu8
                                        reluLayer
                                        
                                        fullyConnectedLayer(2,...
                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                        softmaxLayer
                                        classificationLayer];
                                else if indN==11
                                        layers = [
                                            image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                            
                                            % Conv1
                                            convolution3dLayer([3 3 3],20,...
                                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu1
                                            reluLayer
                                            
                                            % ConvPool2
                                            convolution3dLayer([3 3 3],poolnum,...
                                            'Stride',[2 2 2],'Padding',[1 1 1; 1 1 1],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu2
                                            reluLayer
                                            
                                            % Conv3
                                            convolution3dLayer([1 1 3],35,...
                                            'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu3
                                            reluLayer
                                            
                                            % ConvPool4
                                            convolution3dLayer([1 1 2],poolnum,...
                                            'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu4
                                            reluLayer
                                            
                                            % Conv5
                                            convolution3dLayer([1 1 3],35,...
                                            'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu5
                                            reluLayer
                                            
                                            % ConvPool6
                                            convolution3dLayer([1 1 2],poolnum,...
                                            'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu6
                                            reluLayer
                                            
                                            % Conv7
                                            convolution3dLayer([1 1 3],35,...
                                            'Stride',[1 1 1],'Padding',[0 0 0; 0 0 0],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu7
                                            reluLayer
                                            
                                            % ConvPool8
                                            convolution3dLayer([1 1 2],poolnum,...
                                            'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu8
                                            reluLayer
                                            
                                            % Conv9
                                            convolution3dLayer([1 1 3],35,...
                                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu9
                                            reluLayer
                                            
                                            % ConvPool10
                                            convolution3dLayer([1 1 3],poolnum,...
                                            'Stride',[2 2 2],'Padding',[0 0 1; 0 0 1],...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            
                                            % Relu10
                                            reluLayer
                                            
                                            fullyConnectedLayer(2,...
                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                            softmaxLayer
                                            classificationLayer];
                                    else if indN==10
                                            layers = [
                                                image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                                
                                                % Conv1
                                                convolution3dLayer([3 3 3],20,...
                                                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu1
                                                reluLayer
                                                
                                                % ConvPool2
                                                convolution3dLayer([1 1 3],poolnum,...
                                                'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu2
                                                reluLayer
                                                
                                                % Conv3
                                                convolution3dLayer([1 1 3],35,...
                                                'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu3
                                                reluLayer
                                                
                                                % ConvPool4
                                                convolution3dLayer([1 1 2],poolnum,...
                                                'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu4
                                                reluLayer
                                                
                                                % Conv5
                                                convolution3dLayer([1 1 3],35,...
                                                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu5
                                                reluLayer
                                                
                                                % ConvPool6
                                                convolution3dLayer([1 1 1],poolnum,...
                                                'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu6
                                                reluLayer
                                                
                                                % Conv7
                                                convolution3dLayer([1 1 3],35,...
                                                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu7
                                                reluLayer
                                                
                                                % ConvPool8
                                                convolution3dLayer([1 1 1],poolnum,...
                                                'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu8
                                                reluLayer
                                                
                                                % Conv9
                                                convolution3dLayer([1 1 3],35,...
                                                'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu9
                                                reluLayer
                                                
                                                % ConvPool10
                                                convolution3dLayer([1 1 1],poolnum*2,...
                                                'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                
                                                % Relu10
                                                reluLayer
                                                
                                                fullyConnectedLayer(2,...
                                                'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                softmaxLayer
                                                classificationLayer];
                                        else if indN==8
                                                layers = [
                                                    image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                                    
                                                    % Conv1
                                                    convolution3dLayer([3 3 3],20,...
                                                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu1
                                                    reluLayer
                                                    
                                                    % ConvPool2
                                                    convolution3dLayer([3 3 3],poolnum,...
                                                    'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu2
                                                    reluLayer
                                                    
                                                    % Conv3
                                                    convolution3dLayer([3 3 3],35,...
                                                    'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu3
                                                    reluLayer
                                                    
                                                    % ConvPool4
                                                    convolution3dLayer([1 1 2],poolnum,...
                                                    'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu4
                                                    reluLayer
                                                    
                                                    % Conv5
                                                    convolution3dLayer([1 1 3],35,...
                                                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu5
                                                    reluLayer
                                                    
                                                    % ConvPool6
                                                    convolution3dLayer([1 1 1],poolnum,...
                                                    'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu6
                                                    reluLayer
                                                    
                                                    % Conv7
                                                    convolution3dLayer([1 1 3],35,...
                                                    'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu7
                                                    reluLayer
                                                    
                                                    % ConvPool8
                                                    convolution3dLayer([1 1 1],poolnum*2,...
                                                    'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    
                                                    % Relu8
                                                    reluLayer
                                                    
                                                    fullyConnectedLayer(2,...
                                                    'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                    'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                    softmaxLayer
                                                    classificationLayer];
                                            else if indN==13
                                                    layers = [
                                                        image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                                        
                                                        % Conv1
                                                        convolution3dLayer([3 3 3],20,...
                                                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu1
                                                        reluLayer
                                                        
                                                        % ConvPool2
                                                        convolution3dLayer([3 3 3],poolnum,...
                                                        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu2
                                                        reluLayer
                                                        
                                                        % Conv3
                                                        convolution3dLayer([3 3 3],35,...
                                                        'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu3
                                                        reluLayer
                                                        
                                                        % ConvPool4
                                                        convolution3dLayer([1 1 2],poolnum,...
                                                        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu4
                                                        reluLayer
                                                        
                                                        % Conv5
                                                        convolution3dLayer([1 1 3],35,...
                                                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu5
                                                        reluLayer
                                                        
                                                        % ConvPool6
                                                        convolution3dLayer([1 1 1],poolnum,...
                                                        'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu6
                                                        reluLayer
                                                        
                                                        % Conv7
                                                        convolution3dLayer([1 1 3],35,...
                                                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu7
                                                        reluLayer
                                                        
                                                        % ConvPool8
                                                        convolution3dLayer([1 1 1],poolnum*2,...
                                                        'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu8
                                                        reluLayer
                                                        
                                                        % Conv9
                                                        convolution3dLayer([1 1 3],35,...
                                                        'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu9
                                                        reluLayer
                                                        
                                                        % ConvPool10
                                                        convolution3dLayer([1 1 1],poolnum*2,...
                                                        'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        
                                                        % Relu10
                                                        reluLayer
                                                        
                                                        fullyConnectedLayer(2,...
                                                        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                        'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                        softmaxLayer
                                                        classificationLayer];
                                                else if indN==14
                                                        layers = [
                                                            image3dInputLayer([ptch(2) ptch(1) size(M{1},3)])
                                                            
                                                            % Conv1
                                                            convolution3dLayer([3 3 3],20,...
                                                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu1
                                                            reluLayer
                                                            
                                                            % ConvPool2
                                                            convolution3dLayer([1 1 3],poolnum,...
                                                            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu2
                                                            reluLayer
                                                            
                                                            % Conv3
                                                            convolution3dLayer([3 3 3],35,...
                                                            'Stride',[1 1 1],'Padding',[1 1 1; 1 1 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu3
                                                            reluLayer
                                                            
                                                            % ConvPool4
                                                            convolution3dLayer([1 1 3],poolnum,...
                                                            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu4
                                                            reluLayer
                                                            
                                                            % Conv5
                                                            convolution3dLayer([3 3 3],35,...
                                                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu5
                                                            reluLayer
                                                            
                                                            % ConvPool6
                                                            convolution3dLayer([1 1 2],poolnum,...
                                                            'Stride',[1 1 2],'Padding',[0 0 1; 0 0 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu6
                                                            reluLayer
                                                            
                                                            % Conv7
                                                            convolution3dLayer([1 1 3],35,...
                                                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu7
                                                            reluLayer
                                                            
                                                            % ConvPool8
                                                            convolution3dLayer([1 1 1],poolnum*2,...
                                                            'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu8
                                                            reluLayer
                                                            
                                                            % Conv9
                                                            convolution3dLayer([1 1 3],35,...
                                                            'Stride',[1 1 1],'Padding',[0 0 1; 0 0 1],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu9
                                                            reluLayer
                                                            
                                                            % ConvPool10
                                                            convolution3dLayer([1 1 1],poolnum*2,...
                                                            'Stride',[2 2 2],'Padding',[0 0 0; 0 0 0],...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            
                                                            % Relu10
                                                            reluLayer
                                                            
                                                            fullyConnectedLayer(2,...
                                                            'WeightLearnRateFactor',1,'WeightL2Factor',1,...
                                                            'BiasLearnRateFactor',2,'BiasL2Factor',0)
                                                            softmaxLayer
                                                            classificationLayer];
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

% Calculation
% augimds = augmentedImageDatastore([ptch ptch size(M{1},3)],Xtrain,Ytrain);
% [net,info] = trainNetwork(augimds,layers,options);
tic
[net,info] = trainNetwork(table(Xtrainc,Ytrain),layers,options);
t=toc;
Res.time=t;

% delete(poolobj)

YPred=[];C=[];
% Train
YPred=zeros(length(numimtrain),1);
for j=1:length(numimtrain)
    YPred(j,1) = classify(net,Xtrainc{j});
end
C = confusionmat(Ytrain,categorical(YPred)); % Confusion matrix
Res.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
Res.Train.Recall_T=C(1,1)/sum(C(1,:));
Res.Train.Selectivity_T=C(2,2)/sum(C(2,:));
Res.Train.Precision_T=C(1,1)/sum(C(:,1));
Res.Train.Fmeasure_T=2*Res.Train.Precision_T*Res.Train.Recall_T/(Res.Train.Recall_T+Res.Train.Precision_T);

% Test
idx=[];YPred=[];C=[];
YPred=zeros(length(numimtest),1);
for j=1:length(numimtest)
    YPred(j,1) = classify(net,Xtestc{j});
end
C = confusionmat(Ytest,categorical(YPred)); % Confusion matrix
Res.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
Res.Test.Recall_T=C(1,1)/sum(C(1,:));
Res.Test.Selectivity_T=C(2,2)/sum(C(2,:));
Res.Test.Precision_T=C(1,1)/sum(C(:,1));
Res.Test.Fmeasure_T=2*Res.Test.Precision_T*Res.Test.Recall_T/(Res.Test.Recall_T+Res.Test.Precision_T);

for i=1:nM
    YPred=[];C=[];
    % Train
    idx=find(numimtrain==i);
    YPred=zeros(length(idx),1);
    for j=1:length(idx)
        YPred(j,1) = classify(net,Xtrainc{idx(j)});
    end
    C = confusionmat(Ytrain(idx),categorical(YPred)); % Confusion matrix
    Res.Train.Accuracy(i)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.Train.Recall(i)=C(1,1)/sum(C(1,:));
    Res.Train.Selectivity(i)=C(2,2)/sum(C(2,:));
    Res.Train.Precision(i)=C(1,1)/sum(C(:,1));
    Res.Train.Fmeasure(i)=2*Res.Train.Precision(i)*Res.Train.Recall(i)/(Res.Train.Recall(i)+Res.Train.Precision(i));
    
    % Test
    idx=[];YPred=[];C=[];
    idx=find(numimtest==i);
    YPred=zeros(length(idx),1);
    for j=1:length(idx)
        YPred(j,1) = classify(net,Xtestc{idx(j)});
    end
    C = confusionmat(Ytest(idx),categorical(YPred)); % Confusion matrix
    Res.Test.Accuracy(i)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.Test.Recall(i)=C(1,1)/sum(C(1,:));
    Res.Test.Selectivity(i)=C(2,2)/sum(C(2,:));
    Res.Test.Precision(i)=C(1,1)/sum(C(:,1));
    Res.Test.Fmeasure(i)=2*Res.Test.Precision(i)*Res.Test.Recall(i)/(Res.Test.Recall(i)+Res.Test.Precision(i));
end

disp('Training set')
Res.Train
disp('Test set')
Res.Test
end