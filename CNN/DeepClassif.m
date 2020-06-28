function CartlabDeepbr=DeepClassif(M,wl,RGB,Cartlab,Class,Mtest)
% Fonction qui permet de créer un réseau de deep learning pour la classification.
% INPUT :
%           M : Cube de données hyperspectral
%           wl : Longueur d'onde
%           RGB : Image RGB de l'échantillon
%           Cartlab : Carte labelisée de l'échantillon (cf ANNclassifCreateMap1)
%           Class : Nom des classes (Ex : Class={'Flood','Non-Flood'})
%           Mtest(optionnal) : Cube de données hyperspectral à prédire
% OUTPUT :
%           CartlabDeepbr : Carte de classification par un réseau deep (br
%           = basse résolution car décompose l'image initiale en imagette
%           de 10*10 pixels donc perte de résolution).

if size(M,3)==98||size(M,3)==144
    M=M(:,:,15:end-10);
    wl=wl(15:end-10);
end
if nargin>5
    for i=1:length(Mtest)
        if size(Mtest{i},3)==98||size(Mtest{i},3)==144
            Mtest{i}=Mtest{i}(:,:,15:end-10);
        end
    end
end

% Detection des zones labelisées
stats1 = regionprops(Cartlab==1,'BoundingBox');
stats2 = regionprops(Cartlab==2,'BoundingBox');

stats=[stats1; stats2];
lab=[ones(length(stats1),1); ones(length(stats2),1)*2];

for i=1:length(lab)
    dim(i,:)=stats(i).BoundingBox; %x/y/dx/dy
end

Numpxl=10;
% Selection des zones de plus de 10*10 pixels
[idx10,~]=find(dim(:,3)>Numpxl&dim(:,4)>Numpxl);

% Création d'un jeu de données standardisé en 10*10 pixels
iter=1;
for i=1:length(idx10)
    dx=floor(dim(idx10(i),3)/Numpxl);
    dy=floor(dim(idx10(i),4)/Numpxl);
    for j=1:dx
        for k=1:dy
            X(1:Numpxl,1:Numpxl,1:size(M,3),iter)=M(dim(idx10(i),2)+(j-1*Numpxl)+0.5:dim(idx10(i),2)+Numpxl-1+(j-1*Numpxl)+0.5,dim(idx10(i),1)+(k-1*Numpxl)-0.5:dim(idx10(i),1)+Numpxl-1+(k-1*Numpxl)-0.5,:);
            Y{iter,1}=char(Class(lab(idx10(i))));
            iter=iter+1;
        end
    end
end
Y = categorical(Y);

% Création des jeux de cal et val
idx = randperm(size(X,4),round(0.6*size(X,4)));
trainingImages=X;trainingLabels=Y;
testImages = trainingImages(:,:,:,idx);
trainingImages(:,:,:,idx) = [];
testLabels = trainingLabels(idx,:);
trainingLabels(idx) = [];
augimds = augmentedImageDatastore([Numpxl Numpxl size(M,3)],trainingImages,trainingLabels);

numImageCategories = length(unique(Y));

%% Create a Convolutional Neural Network (CNN)
% Create the image input layer for 32x32x3 CIFAR-10 images
[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize);

% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
    
% The first convolutional layer has a bank of 32 5x5x3 filters. A
% symmetric padding of 2 pixels is added to ensure that image borders
% are included in the processing. This is important to avoid
% information at the borders being washed away too early in the
% network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)

% Note that the third dimension of the filter can be omitted because it
% is automatically deduced based on the connectivity of the network. In
% this case because this layer follows the image layer, the third
% dimension must be 3 to match the number of channels in the input
% image.

% Next add the ReLU layer:
reluLayer()

% Follow it with a max pooling layer that has a 3x3 spatial pooling area
% and a stride of 2 pixels. This down-samples the data dimensions from
% 32x32 to 15x15.
maxPooling2dLayer(3, 'Stride', 2)

% Repeat the 3 core layers to complete the middle of the network.
convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

% convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
% reluLayer()
% maxPooling2dLayer(3, 'Stride',2)

];

finalLayers = [
    
% Add a fully connected layer with 64 output neurons. The output size of
% this layer will be an array with a length of 64.
fullyConnectedLayer(64)

% Add an ReLU non-linearity.
reluLayer

% Add the last fully connected layer. At this point, the network must
% produce numImageCategories signals that can be used to measure whether the input image
% belongs to one category or another. This measurement is made using the
% subsequent loss layers.
fullyConnectedLayer(numImageCategories)

% Add the softmax loss layer and classification layer. The final layers use
% the output of the fully connected layer to compute the categorical
% probability distribution over the image classes. During the training
% process, all the network weights are tuned to minimize the loss over this
% categorical distribution.
softmaxLayer
%classificationLayer
pixelClassificationLayer
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

%% Train CNN
% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true, ...
    'Plots','training-progress',...
    'ValidationData',{testImages,testLabels});

% Train a network.
cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);

%% Validate Network Training
% % Extract the first convolutional layer weights
% w = cifar10Net.Layers(2).Weights;
%
% % rescale the weights to the range [0, 1] for better visualization
% w = rescale(w);
%
% nbfig=ceil(size(w,3)/20);
% iter=1;
% for j=1:nbfig
%     figure
%     for i=1:20
%         if iter<size(w,3)
%             subplot(4,5,i)
%             montage(squeeze(w(:,:,iter,:)))
%             colormap(jet)
%             colorbar
%             title(strcat(num2str(wl(iter)),' nm'))
%         end
%         iter=iter+1;
%     end
% end

% Run the network on the test set.
YTest = classify(cifar10Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels);

%% Test imagettes
if nargin<6
    % Decoupe l'image en imagette de 10*10
    ix=floor(size(M,1)/Numpxl);
    iy=floor(size(M,2)/Numpxl);
    iter=1;
    for i=1:ix
        for j=1:iy
            Mi(1:Numpxl,1:Numpxl,:,iter)=M(1+(i-1)*Numpxl:Numpxl+(i-1)*Numpxl,1+(j-1)*Numpxl:Numpxl+(j-1)*Numpxl,:);
            iter=iter+1;
        end
    end
    % Detect class
    YPred = predict(cifar10Net,Mi); % Prédit les imagettes donc pertes de résolutions
    [Yp,~]=find(YPred'==max(YPred'));
    
    % Creation de la carte de classif
    iter=1;
    for i=1:ix
        for j=1:iy
            CartlabDeepbr(i,j)=Yp(iter);
            iter=iter+1;
        end
    end
    
else
    for k=1:length(Mtest)
        % Decoupe l'image en imagette de 10*10
        ix=floor(size(Mtest{k},1)/Numpxl);
        iy=floor(size(Mtest{k},2)/Numpxl);
        iter=1;
        Mi=[];
        for i=1:ix
            for j=1:iy
                Mi(1:Numpxl,1:Numpxl,:,iter)=Mtest{k}(1+(i-1)*Numpxl:Numpxl+(i-1)*Numpxl,1+(j-1)*Numpxl:Numpxl+(j-1)*Numpxl,:);
                iter=iter+1;
            end
        end
        % Detect class
        Ypred=[];Yp=[];
        YPred = predict(cifar10Net,Mi); % Prédit les imagettes donc pertes de résolutions
        [Yp,~]=find(YPred'==max(YPred'));
        
        % Creation de la carte de classif
        iter=1;
        for i=1:ix
            for j=1:iy
                CartlabDeepbr{k}(i,j)=Yp(iter);
                iter=iter+1;
            end
        end
    end
end

%% Comparaison ANN
if nargin<6
    [CartlabANN] = ANNclassifTest(M,wl,RGB,Cartlab,2);
    close all
end

% Sortie graphique
if nargin<6
    figure;
    ha(1)=subplot(411);
    imagesc(imresize(Cartlab,[size(CartlabDeepbr,1) size(CartlabDeepbr,2)]))
    ha(2)=subplot(412);
    imagesc(CartlabDeepbr)
    ha(3)=subplot(413);
    imagesc(imresize(CartlabANN,[size(CartlabDeepbr,1) size(CartlabDeepbr,2)]))
    ha(4)=subplot(414);
    imagesc(imresize(RGB,[size(CartlabDeepbr,1) size(CartlabDeepbr,2)]))
    linkaxes(ha,'xy')
else
    for i=1:length(Mtest)
        figure;
        ha(1)=subplot(211);
        imagesc(CartlabDeepbr{i})
        ha(2)=subplot(212);
        if size(Mtest{i},3)==74
            imagesc(imresize(Mtest{i}(:,:,[35 11 1]),[size(CartlabDeepbr,1) size(CartlabDeepbr,2)]))
        else
            imagesc(imresize(Mtest{i}(:,:,[92 96 109]),[size(CartlabDeepbr,1) size(CartlabDeepbr,2)]))
        end
        linkaxes(ha,'xy')
    end
end
end