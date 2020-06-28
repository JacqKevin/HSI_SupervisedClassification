function ANNclassif(M,IM,M2,IM2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% ROI
roiansw=questdlg('Y a t''il des zones hors échantillons à enlever?','ROI échantillon','Oui','Non','Oui');
if strcmp(roiansw,'Oui')
    figure;
    if nargin<3
        imagesc(IM)
    else
        ha(1)=subplot(211);
        imagesc(IM)
        ha(2)=subplot(212);
        imagesc(IM2)
        linkaxes(ha,'xy')
    end
    hBox = imrect;
    disp('Selectionner un rectangle puis double cliquer dessus')
    roiPosition = wait(hBox);
    
    if nargin>3
        ROI2=M2(roiPosition(2):roiPosition(2)+roiPosition(4),roiPosition(1):roiPosition(1)+roiPosition(3),:);
    end
    if size(ROI2,3)==3
        ROIim2=ROI2;
    else if size(ROI2,3)==144
            ROIim2=ROI2(:,:,[107 111 124])/10000;
        else if size(ROI2,3)==98
                ROIim2=ROI2(:,:,[50 26 13])/10000;
            end
        end
    end
    
    ROI=M(roiPosition(2):roiPosition(2)+roiPosition(4),roiPosition(1):roiPosition(1)+roiPosition(3),:);
    if size(ROI,3)==3
        ROIim=ROI;
    else if size(ROI,3)==144
            ROIim=ROI(:,:,[107 111 124])/10000;
        else if size(ROI,3)==98
                ROIim=ROI(:,:,[50 26 13])/10000;
            end
        end
    end
else
    ROI=M;
    if size(ROI,3)==3
        ROIim=ROI;
    else if size(ROI,3)==144
            ROIim=ROI(:,:,[107 111 124])/10000;
        else if size(ROI,3)==98
                ROIim=ROI(:,:,[50 26 13])/10000;
            end
        end
    end
end
nbpixel=size(ROI,1)*size(ROI,2);

if size(ROI,3)==98||size(ROI,3)==144
    ROI=ROI(:,:,15:end-10);
end

ROId=reshape(ROI,[],size(ROI,3));

if size(ROI2,3)==98||size(ROI2,3)==144
    ROI2=ROI2(:,:,15:end-10);
end

ROId2=reshape(ROI2,[],size(ROI2,3));

% Selection des lamines
ROIclass=[];class=[];Mclass=[];
Cartlab=zeros(size(ROI,1),size(ROI,2));
disp('Selection des lamines pour classification supervisée')
roisearch='Oui';
list = {'1','2','3','4','5'};

figure;
if nargin<3
    imagesc(ROIim)
else
    ha(1)=subplot(211);
    imagesc(ROIim)
    ha(2)=subplot(212);
    imagesc(ROIim2)
    linkaxes(ha,'xy')
end

while strcmp(roisearch,'Oui')
    hBox = imrect;
    roiPositioni = wait(hBox);
    ROIid=reshape(ROIim(roiPositioni(2):roiPositioni(2)+roiPositioni(4),roiPositioni(1):roiPositioni(1)+roiPositioni(3),:),[],size(ROIim,3));
        
    Mid=reshape(ROI(roiPositioni(2):roiPositioni(2)+roiPositioni(4),roiPositioni(1):roiPositioni(1)+roiPositioni(3),:),[],size(ROI,3));
    if nargin>2
        Mid2=reshape(ROI2(roiPositioni(2):roiPositioni(2)+roiPositioni(4),roiPositioni(1):roiPositioni(1)+roiPositioni(3),:),[],size(ROI2,3));
    end
    
    [indx,~] = listdlg('ListString',list);
    if isempty(find(class==indx))
        lab=inputdlg({strcat('Nom de la classe ',num2str(indx))},'Labelisation');
        lbl{indx}=lab{1,1};
    end
    class=[class; ones(size(ROIid,1),1)*indx];
    Cartlab(roiPositioni(2):roiPositioni(2)+roiPositioni(4),roiPositioni(1):roiPositioni(1)+roiPositioni(3))=indx;
    nbclassi(indx)=length(find(class==indx));
    ROIclass=[ROIclass; ROIid];
    
    Mclass=[Mclass;Mid];
    if nargin>2
        Mclass2=[Mclass;Mid2];
    end
    list{indx}=strcat(lbl{indx},' : ',num2str(nbclassi(indx)/nbpixel),'%');
    
    roisearch=questdlg('Voulez vous continuez à sélectionner des zones?','ROI sélection','Oui','Non','Oui');
end

nbclass=length(unique(class));
classi=unique(class);

classwork=class;

ROIclasswork=Mclass;
if nargin>2
    ROIclasswork2=Mclass2;
end
indxclasswork=classi;

classworkk=zeros(size(classwork,1),length(indxclasswork));
for i=1:size(classwork,1)
    classworkk(i,classwork(i,1))=1;
end

%% Pattern Recognition ANN
% Data
x = ROIclasswork';
t = classworkk';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainParam.showWindow = false;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotconfusion'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% Plots
figure, plotconfusion(t,y)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end

if nargin>2
    x = ROIclasswork2';
    t = classworkk';
    
    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
    
    % Create a Pattern Recognition Network
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    
    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivision
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = false;
    
    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'crossentropy';  % Cross-Entropy
    
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotconfusion'};
    
    % Train the Network
    [net,tr] = train(net,x,t);
    
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    
    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y);
    valPerformance = perform(net,valTargets,y);
    testPerformance = perform(net,testTargets,y);
    
    % Plots
    figure, plotconfusion(t,y)
    
    % Deployment
    % Change the (false) values to (true) to enable the following code blocks.
    % See the help for each generation function for more information.
    if (false)
        % Generate MATLAB function for neural network for application
        % deployment in MATLAB scripts or with MATLAB Compiler and Builder
        % tools, or simply to examine the calculations your trained neural
        % network performs.
        genFunction(net,'myNeuralNetworkFunction2');
        y = myNeuralNetworkFunction2(x);
    end
end
if (true)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    name=inputdlg('Nom du modèle développé','Modèle');
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    classROId = myNeuralNetworkFunction(ROId');
    
    if nargin>2
        genFunction(net,'myNeuralNetworkFunction2','MatrixOnly','yes');
        classROId2 = myNeuralNetworkFunction(ROId2');
    end
    for k=1:size(classROId,1)
        classROI=reshape(classROId(k,:),size(ROI,1),size(ROI,2));
        if nargin>2
            classROI2=reshape(classROId2(k,:),size(ROI,1),size(ROI,2));
        end
        
        figure;
        if nargin<3
            ha(1)=subplot(211);imagesc(classROI),colorbar,colormap(jet)
            ha(2)=subplot(212);imagesc(ROIim),colorbar
        else
            ha(1)=subplot(411);imagesc(classROI),colorbar,colormap(jet)
            ha(2)=subplot(412);imagesc(ROIim),colorbar
            ha(3)=subplot(414);imagesc(classROI2),colorbar
            ha(4)=subplot(414);imagesc(ROIim2),colorbar
        end
        linkaxes(ha,'xy')
        
    end
    movefile('myNeuralNetworkFunction.m',strcat(name{1,1},'.m'));
    
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end

assignin('base','Cartlab',Cartlab)
assignin('base','Mlab',ROI)
end