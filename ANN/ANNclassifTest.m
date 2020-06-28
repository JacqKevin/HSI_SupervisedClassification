function [CartlabANN] = ANNclassifTest(M,wl,RGB,Cartlab,hiddenLayerSize,name)
% Function to estime a discrimination model with an artificial neural
% network.

% INPUT:
%           M: Hyperspectral dataset
%           wl: Wavelengths
%           RGB: Image of the sample
%           Cartlab: Labelled map
%           hiddenLayerSize: Number of neurons
%           name: Name of the project for graphical and data saving output
% OUTPUT:
%           CartlabANN: Classification map

if nargin<5
    hiddenLayerSize=10;
end

if size(M,3)==98||size(M,3)==144
    M=M(:,:,15:end-10);
    wl=wl(15:end-10);
end

ROId=reshape(M,[],size(M,3));

%% superpixel
spp=questdlg('Do you want to compute threepixels?');
if strcmp(spp,'Yes')
    Mspp=zeros(size(M,1),size(M,2),size(M,3)*2);
    for i=2:size(M,1)
        for j=1:size(M,2)
            Mspp(i,j,:)=[squeeze(M(i-1,j,:)); squeeze(M(i,j,:))]';
        end
    end
end

%%
[a,~]=find(reshape(Cartlab,[],1)>0);

Cartlabd=reshape(Cartlab,[],1);
Md=reshape(M,[],size(M,3));

classworkk=Cartlabd(a,:);
ROIclasswork=Md(a,:);

if size(classworkk,2)==1
    classworkki=zeros(length(classworkk),max(classworkk));
    for i=1:length(classworkk)
        classworkki(i,classworkk(i))=1;
    end
else
    classworkki=classworkk;
end

%% ANN Pattern Recognition 
% Data
x = ROIclasswork';
t = classworkki';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
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
saveas(gca,'Confusion.jpg')

% Weight
figure;
plot(wl,cell2mat(net.IW)),
grid on,
xlim([wl(1),wl(end)])
xlabel('Wavelength (nm)')
ylabel('Weight')
saveas(gca,'Weights.jpg')

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

if (true)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    classROId = myNeuralNetworkFunction(ROId');
    
    [a,~]=find(classROId==max(classROId));
    CartlabANN=reshape(a,size(M,1),size(M,2));
    
    figure;
    ha(1)=subplot(211);imagesc(CartlabANN),colorbar,colormap(jet)
    ha(2)=subplot(212);imagesc(RGBrehaussee(RGB)),colorbar
    linkaxes(ha,'xy')
    
    if nargin>5
        movefile('myNeuralNetworkFunction.m',strcat(name,'.m'));
        movefile('Confusion.jpg',strcat(name, '_Confusion.jpg'));
        movefile('Weights.jpg',strcat(name,'_Weights.jpg'));
    end
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end

end