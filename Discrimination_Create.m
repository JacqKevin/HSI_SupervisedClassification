function [Mdl,IM, IMlabel] =Discrimination_Create(M,RGB,d,wl,Cartlab,algo,fig)
% Function to estimate classification models with DT, RF, ANN, 
% CNN, LDA, QDA, PLS-DA approaches.
%
% INPUT:
%       M: HSI datacube (n*m*p)
%       RGB: Image of the sample
%       d : Associated depth (1*m)(mm)
%       wl: Associated wavelength (1*p)
%       Cartlab: Labelled map (n*m) or class vector (z*k)
%		associated with its depth vector (z*1-2)
%       algo: Discrimination algorithm to use
%		0: all
%		1: Decision tree DT
%		2: Random forest RF
%		3: Artificial neural network ANN
%		4: Convolutional neural network CNN
%		5: Linear discriminant analysis LDA
%		6: Quadratic discriminant analysis QDA
%		7: Partial least squares discriminant analysis PLS-DA
%       fig: Display figure with value different from 0
% OUTPUT:
%       Mdl: Discrimination models
%       IM: Predicted classification maps
%       IMlabel: Name of the variable predicted
%
% This is the Matalb toolbox from the papers:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Jacq, K., Rapuc, W., Benoit, A., Develle, A.-L., Coquin, D., 
% Fanget, B., Perrette, Y.,  Sabatier, P., Wilhelm, B., 
% Debret, M., Arnaud, F., 2019. Sedimentary structures 
% discriminations with hyperspectral imaging on 
% sediment cores. Computers & Geosciences

% Please cite our papers if you use our code for your research.

Mdl.Method='Classification';
Mdlvs.Method='Classification';
Mdl.Type='Complete';
Mdlvs.Type='Reduce';

if nargin<5
    algo=0;
    fig=0;
end
if nargin<6
    fig=0;
end
if sum(algo==0)==1
    numalgo=16;
else if sum(algo==3)>=1
        numalgo=9+length(algo)-1;
    else
        numalgo=length(algo);
    end
end

m=median(M(:));
if m>1000
    M=M/10000;
end

% Parameters
if size(M,3)==98||size(M,3)==144
    Mdl.idx_wl1=15:size(M,3)-10;
    Mdlvs.idx_wl1=15:size(M,3)-10;
    M=M(:,:,15:end-10);
    wl=wl(15:end-10);
end

% Moisture band with NIR data
if mean(wl)>1000
    water = questdlg('Do you want to remove moisture bands ?','Moisture','Yes','No','No');
    if strcmp(water,'Yes')
        z1=[1094 1176];
        z2=[1339 1466];
        z3=[1768 2005];
        idxz1=wl>z1(1)&wl<z1(2);
        idxz2=wl>z2(1)&wl<z2(2);
        idxz3=wl>z3(1)&wl<z3(2);
        idxr=idxz1+idxz2+idxz3;
        idxr=idxr>0;% bandes enlevées
    else
        idxr=zeros(1,length(wl));
    end
else
    idxr=zeros(1,length(wl));
end
Mdl.idx_wl2=idxr;
Mdlvs.idx_wl2=idxr;

% Map labelled or class vector
if iscell(Cartlab)
    Y=Cartlab{1};
    dY=Cartlab{2};
    clear Cartlab
end

% Conversion and normalization
pseudoabs = questdlg('Do you want to convert your data in pseudo-absorbance ?','Pseudo-Absorbance','Yes','No','No');
datapretreat = questdlg('Do you want to center or normalized your data?','Preprocessing','No','Center','Autoscaling','No');

if strcmp(pseudoabs,'Yes')
    Mdl.Spectral='Absorbance';
    Mdlvs.Spectral='Absorbance';
else
    Mdl.Spectral='Reflectance';
    Mdlvs.Spectral='Reflectance';
end
Mdl.Preprocessing=datapretreat;
Mdlvs.Preprocessing=datapretreat;

%% Conversion in pseudo-absorbance
if strcmp(pseudoabs,'Yes')
    S=reshape(M,size(M,1)*size(M,2),size(M,3));
    [a,b]=find(S==0);
    S(a,b)=eps;
    S=log(1./S);
    M=reshape(S,size(M,1),size(M,2),size(M,3));
end

%% Spectral preprocessing
datapretreat2 = questdlg('Do you want to preprocess the spectra ?','Preprocess','Yes','No','No');
if strcmp(datapretreat2,'Yes')
    pret={'Detrend','SNV','SNVD','MSC','D1','D2','SNV+D1','SNVD+D1','MSC+D1','SNV+D2','SNVD+D2','MSC+D2','CR'};
    datapretreat2b = listdlg('ListString',pret);
    M = reshape(AllPret(reshape(M,[],size(M,3)),wl,datapretreat2b),size(M,1),size(M,2),size(M,3));
    Mdl.SpectralProcessing=pret{datapretreat2b};
    Mdlvs.SpectralProcessing=pret{datapretreat2b};
else
    Mdl.SpectralProcessing='Raw';
    Mdlvs.SpectralProcessing='Raw';
end

%% Normalization-Centering
if strcmp(datapretreat,'Center')
    [Stmp,para1,para2]=pretreat(reshape(M,[],size(M,3))','center');
    Mdl.PreprocessingPara.para1=para1;
    Mdl.PreprocessingPara.para2=para2;
    Mdlvs.PreprocessingPara.para1=para1;
    Mdlvs.PreprocessingPara.para2=para2;
    M=reshape(Stmp',size(M,1),size(M,2),size(M,3));
    clear Stmp
else if strcmp(datapretreat,'Autoscaling')
        [Stmp,para1,para2]=pretreat(reshape(M,[],size(M,3))','autoscaling');
        Mdl.PreprocessingPara.para1=para1;
        Mdl.PreprocessingPara.para2=para2;
        Mdlvs.PreprocessingPara.para1=para1;
        Mdlvs.PreprocessingPara.para2=para2;
        M=reshape(Stmp',size(M,1),size(M,2),size(M,3));
        clear Stmp
    end
end

if exist('Cartlab','var')% Labelled map
    itermi=0;
    label_train=[];
    if fig==1
        % Display labelled map
        figure;
        ha(1)=subplot(211);
        imagesc(RGB*(0.5/mean(RGB(:))))
        ha(2)=subplot(212);
        imagesc(Cartlab);
        linkaxes(ha,'xy')
    end
    
    % Developped
    Cartlabd=reshape(Cartlab,[],1);
    Md=reshape(M,[],size(M,3));
    
    % Find the number of labeled classes
    m=max(Cartlab(:));
    
    % Subsampling of the image (5*5*channel*nb patch)
    Xtp=[];Ytp=[];
    Xtp=Md(Cartlabd>0,:);
    Ytp=Cartlabd(Cartlabd>0,:);
    %     itermi=itermi+size(areas,1);
    
    % Set definition with same size in each class
    [C,~,~] = unique(Ytp); % number of class
    for i=1:length(C)
        numc(i)=sum(Ytp==C(i)); % number per class
    end
    cal=round(0.7*min(numc));  % 70% in the training set
    for i=1:length(C)
        al = rand(1,numc(i)); % random number
        [~,iii] = sort(al);
        [tp,~]=find(Ytp==C(i));
        if iM==1&&i==1
            Xtrain=Xtp(tp(iii(1:cal)),:);
            Xtest=Xtp(tp(iii(cal+1:end)),:);
            Ytrain=Ytp(tp(iii(1:cal)),1);
            Ytest=Ytp(tp(iii(cal+1:end)),1);
        else
            Xtrain=cat(1,Xtrain,Xtp(tp(iii(1:cal)),:));
            Xtest=cat(1,Xtest,Xtp(tp(iii(cal+1:end)),:));
            Ytrain=[Ytrain; Ytp(tp(iii(1:cal)),1)];
            Ytest=[Ytest; Ytp(tp(iii(cal+1:end)),1)];
        end
    end
    clear numtp Xtp Md Cartlabd
else % Vector class
    dataref = questdlg('How your thickness are calculated?','Reference thickness','Min','Center','Max','Max');
    if size(dY,2)==1
        datadref = inputdlg('What is the thickness:');
        datadref=repmat(str2double(datadref),1,length(d));
    else
        datadref=dY(:,2);
        dY=dY(:,1);
    end
    ech = questdlg('How would you like to subsample?','Subsampling','Random selection','Mean','Median','Random selection');
    
    % Create Reference map
    IMref=zeros(size(M,1),size(M,2));
    for i=1:length(Y)
        if strcmp(dataref,'Max')
            [~,a]=find(abs(d-(dY(i)-datadref(i)))==min(abs(d-(dY(i)-datadref(i)))));
            [~,b]=find(abs(d-(dY(i)))==min(abs(d-(dY(i)))));
        else if strcmp(dataref,'Min')
                [~,a]=find(abs(d-(dY(i)))==min(abs(d-(dY(i)))));
                [~,b]=find(abs(d-(dY(i)+datadref(i)))==min(abs(d-(dY(i)+datadref(i)))));
            else
                [~,a]=find(abs(d-(dY(i)-0.5*datadref(i)))==min(abs(d-(dY(i)-0.5*datadref(i)))));
                [~,b]=find(abs(d-(dY(i)+0.5*datadref(i)))==min(abs(d-(dY(i)+0.5*datadref(i)))));
            end
        end
        IMref(:,a:b)=Y(i);
    end
    
    %% Subsampling of the hyperspectral data at the reference resolution
    dyb=zeros(2,length(dY));
    if strcmp(dataref,'Max')
        for i=1:length(dY)
            dyb(1,i)=dY(i)-2/3*datadref(i);
            dyb(2,i)=dY(i)-1/3*datadref(i);
            dyc(1,i)=dY(i)-datadref(i);
            dyc(2,i)=dY(i);
        end
    else if strcmp(dataref,'Min')
            for i=1:length(dY)
                dyb(1,i)=dY(i)+1/3*datadref(i);
                dyb(2,i)=dY(i)+2/3*datadref(i);
                dyc(1,i)=dY(i);
                dyc(2,i)=dY(i)+datadref(i);
            end
        else if strcmp(dataref,'Center')
                for i=1:length(dY)
                    dyb(1,i)=dY(i)-1/6*datadref(i);
                    dyb(2,i)=dY(i)+1/6*datadref(i);
                    dyc(1,i)=dY(i)-1/2*datadref(i);
                    dyc(2,i)=dY(i)+1/2*datadref(i);
                end
            end
        end
    end
    
    % Depth registration
    iter=1;
    diy=zeros(2,length(dY));
    for i=1:length(dY)
        for j=1:2
            [~, a]=find(abs(d-dyb(j,i))==min(abs(d-dyb(j,i))));
            if abs(d(a(1))-dyb(j,i))<0.5
                dis(j,iter)=a(1);
                Ye(iter,:)=Y(i,:);
            else
                diy(j)=1;
            end
            
            [~, b]=find(abs(d-dyc(j,i))==min(abs(d-dyc(j,i))));
            disc(j,i)=b(1);
        end
        if abs(d(a(1))-dyb(j,i))<0.5
            iter=iter+1;
        end
    end
    if dis(2,end)==0
        dis=dis(:,1:end-1);
        Ye=Ye(1:end-1,:);
    end
    
    % Keep the central third
    Mi=M(round(1/3*size(M,1):2/3*size(M,1)),:,:);
    Mia=M(round(1/3*size(M,1):2/3*size(M,1)),:,:);
    
    disx=dis(2,:)-dis(1,:);
    if sum(disx==0)>0
        a=disx==0;
        disx=disx(1,a==0);
        Ye=Ye(a==0,:);
    end
    if sum(disx<0)>0
        [a,b]=find(disx<0);
        for i=1:length(b)
            disxi= dis(1,b(i));
            dis(1,b(i))=dis(2,b(i));
            dis(2,b(i))=disxi;
        end
    end
    % coordonnée y représente le nombre de colonne du plan :
    disy=ones(1,size(Ye,1))*size(Mi,1);
    
    % Subsampling methods
    Sw=[];
    if strcmp(ech,'Random selection')
        nbiter=100;
        for i = 1 : nbiter
            tiragex=ceil(rand(1,size(Ye,1)).*disx);
            tiragey=ceil(rand(1,size(Ye,1)).*disy);
            
            for j = 1:size(Ye,1)
                
                        Sw(i,j,:)=squeeze(Mi(tiragey(j),dis(j)-1+tiragex(j),:));
                Sboot(i,j,:)=Mia(tiragey(j),dis(j)-1+tiragex(j),:);
            end
        end
    else if strcmp(ech,'Mean')
            for j = 1:size(Ye,1)
                Mii=[];
                Mii=Mi(:,dis(1,j):dis(2,j),:);
                Si=reshape(Mii,[size(Mii,1)*size(Mii,2) size(Mii,3)]);
                % Remove outliers
                [~,outliers] = ACPauto(Si, 0, wl);
                Si=Si(outliers(:,7)==0,:);
                % Remove outliers
                [~,outliers] = ACPauto(Si, 0, wl);
                Si=Si(outliers(:,7)==0,:);
                Sw(1,j,:)=mean(Si);
                Yee(j,:)=Ye(j,:);
            end
            nbiter=1;
        else if strcmp(ech,'Median')
                for j = 1:size(Ye,1)
                    Mii=[];
                    Mii=Mi(:,dis(1,j):dis(2,j),:);
                    Si=reshape(Mii,[size(Mii,1)*size(Mii,2) size(Mii,3)]);
                    % Remove outliers
                    [~,outliers] = ACPauto(Si, 0, wl);
                    Si=Si(outliers(:,7)==0,:);
                    % Remove outliers
                    [~,outliers] = ACPauto(Si, 0, wl);
                    Si=Si(outliers(:,7)==0,:);
                    Sw(1,j,:)=median(Si);
                    Yee(j,:)=Ye(j,:);
                end
                nbiter=1;
            end
        end
    end
    Sboot=squeeze(Sw); 
    Sboot=Sboot(:,idxr==0);
    M=M(:,:,idxr==0);
    wl=wl(idxr==0);
    
    % Set definition with same size in each class
    if islogical(Y)
        Y=double(Y);
    end
    Ytp=Y;
    [C,~,~] = unique(Ytp); % number of class
    for i=1:length(C)
        numc(i)=sum(Ytp==C(i)); % number per class
    end
    cal=round(0.7*min(numc));  % 70% in the training set
    if cal<10
        cal=median(round(0.7*numc));
    end
    
    for i=1:length(C)
        al = rand(1,numc(i)); % random number
        [~,iii] = sort(al);
        [tp,~]=find(Ytp==C(i));
        
        if numc(i)>cal
            if i==1
                Xtrain=Sboot(tp(iii(1:cal)),:);
                Xtest=Sboot(tp(iii(cal+1:end)),:);
                Ytrain=Ytp(tp(iii(1:cal)),1);
                Ytest=Ytp(tp(iii(cal+1:end)),1);
                
            else
                Xtrain=cat(1,Xtrain,Sboot(tp(iii(1:cal)),:));
                Xtest=cat(1,Xtest,Sboot(tp(iii(cal+1:end)),:));
                Ytrain=[Ytrain; Ytp(tp(iii(1:cal)),1)];
                Ytest=[Ytest; Ytp(tp(iii(cal+1:end)),1)];
            end
        else
            if i==1
            Xtrain=Sboot(tp(iii(:)),:);
            Xtest=Sboot(tp(iii(:)),:);
            Ytrain=Ytp(tp(iii(:)),1);
            Ytest=Ytp(tp(iii(:)),1);
            else
                Xtrain=cat(1,Xtrain,Sboot(tp(iii(:)),:));
                Xtest=cat(1,Xtest,Sboot(tp(iii(:)),:));
                Ytrain=[Ytrain; Ytp(tp(iii(:)),1)];
                Ytest=[Ytest; Ytp(tp(iii(:)),1)];
            end
        end
        
    end
end
Ytrain=categorical(Ytrain);
Ytest=categorical(Ytest);

if fig==1
    figure;
    subplot(121)
    for j=1:length(C)
        atrain(1,j)=length(Ytrain(Ytrain==categorical(C(j))));
    end
    bar(atrain','stacked')
    grid on
    subplot(122)
    for j=1:length(C)
        atest(1,j)=length(Ytest(Ytest==categorical(C(j))));
    end
    bar(atest','stacked')
    grid on
end

h=waitbar(0);
nalgo=1;
if algo==0|sum(algo==1)==1
    %% DT
    disp('DT')
    waitbar(nalgo/numalgo)
    tic
    Mdl.DT = fitctree(Xtrain,Ytrain);
    t=toc;
    Res.DT.time=t;
    
    label_train = resubPredict(Mdl.DT);
    C = confusionmat(Ytrain,label_train);
    
    Res.DT.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.DT.Train.Recall_T=C(1,1)/sum(C(1,:));
    Res.DT.Train.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.DT.Train.Precision_T=C(1,1)/sum(C(:,1));
    Res.DT.Train.Fmeasure_T=2*Res.DT.Train.Precision_T*Res.DT.Train.Recall_T/(Res.DT.Train.Recall_T+Res.DT.Train.Precision_T);
    
    [label_test,score_test,node_test,cnum_test]  = predict(Mdl.DT,Xtest);
    C = confusionmat(Ytest,label_test);
    
    Res.DT.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.DT.Test.Recall_T=C(1,1)/sum(C(1,:));
    Res.DT.Test.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.DT.Test.Precision_T=C(1,1)/sum(C(:,1));
    Res.DT.Test.Fmeasure_T=2*Res.DT.Test.Precision_T*Res.DT.Test.Recall_T/(Res.DT.Test.Recall_T+Res.DT.Test.Precision_T);
    
    DT=[Res.DT.Train.Accuracy_T, Res.DT.Train.Recall_T, Res.DT.Train.Selectivity_T, Res.DT.Train.Precision_T,Res.DT.Train.Fmeasure_T,0,Res.DT.Test.Accuracy_T, Res.DT.Test.Recall_T, Res.DT.Test.Selectivity_T, Res.DT.Test.Precision_T,Res.DT.Test.Fmeasure_T];
    
    C = confusionmat(Ytrain,label_train);
    Res.DT.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.DT.Train.Recall(1)=C(1,1)/sum(C(1,:));
    Res.DT.Train.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.DT.Train.Precision(1)=C(1,1)/sum(C(:,1));
    Res.DT.Train.Fmeasure(1)=2*Res.DT.Train.Precision(1)*Res.DT.Train.Recall(1)/(Res.DT.Train.Recall(1)+Res.DT.Train.Precision(1));
    
    C = confusionmat(Ytest,label_test);
    Res.DT.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.DT.Test.Recall(1)=C(1,1)/sum(C(1,:));
    Res.DT.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.DT.Test.Precision(1)=C(1,1)/sum(C(:,1));
    Res.DT.Test.Fmeasure(1)=2*Res.DT.Test.Precision(1)*Res.DT.Test.Recall(1)/(Res.DT.Test.Recall(1)+Res.DT.Test.Precision(1));
    
    nalgo=nalgo+1;
end
if algo==0|sum(algo==2)==1
    %% RF
    disp('RF')
    waitbar(nalgo/numalgo)
    tic
    Mdl.RF = fitcensemble(Xtrain,Ytrain);
    t=toc;
    Res.RF.time=t;
    
    label_train = resubPredict(Mdl.RF);
    C = confusionmat(Ytrain,label_train);
    
    Res.RF.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.RF.Train.Recall_T=C(1,1)/sum(C(1,:));
    Res.RF.Train.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.RF.Train.Precision_T=C(1,1)/sum(C(:,1));
    Res.RF.Train.Fmeasure_T=2*Res.RF.Train.Precision_T*Res.RF.Train.Recall_T/(Res.RF.Train.Recall_T+Res.RF.Train.Precision_T);
    
    [label_test,scores_test] = predict(Mdl.RF,Xtest);
    C = confusionmat(Ytest,label_test);
    
    Res.RF.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.RF.Test.Recall_T=C(1,1)/sum(C(1,:));
    Res.RF.Test.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.RF.Test.Precision_T=C(1,1)/sum(C(:,1));
    Res.RF.Test.Fmeasure_T=2*Res.RF.Test.Precision_T*Res.RF.Test.Recall_T/(Res.RF.Test.Recall_T+Res.RF.Test.Precision_T);
    
    RF=[Res.RF.Train.Accuracy_T, Res.RF.Train.Recall_T, Res.RF.Train.Selectivity_T, Res.RF.Train.Precision_T,Res.RF.Train.Fmeasure_T,0,Res.RF.Test.Accuracy_T, Res.RF.Test.Recall_T, Res.RF.Test.Selectivity_T, Res.RF.Test.Precision_T,Res.RF.Test.Fmeasure_T];
    
    C = confusionmat(Ytrain,label_train);
    Res.RF.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.RF.Train.Recall(1)=C(1,1)/sum(C(1,:));
    Res.RF.Train.Select1vity(1)=C(2,2)/sum(C(2,:));
    Res.RF.Train.Precision(1)=C(1,1)/sum(C(:,1));
    Res.RF.Train.Fmeasure(1)=2*Res.RF.Train.Precision(1)*Res.RF.Train.Recall(1)/(Res.RF.Train.Recall(1)+Res.RF.Train.Precision(1));
    
    C = confusionmat(Ytest,label_test);
    Res.RF.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.RF.Test.Recall(1)=C(1,1)/sum(C(1,:));
    Res.RF.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.RF.Test.Precision(1)=C(1,1)/sum(C(:,1));
    Res.RF.Test.Fmeasure(1)=2*Res.RF.Test.Precision(1)*Res.RF.Test.Recall(1)/(Res.RF.Test.Recall(1)+Res.RF.Test.Precision(1));
    
    nalgo=nalgo+1;
end
if algo==0|sum(algo==3)==1
    %% ANN
    %
    Ytrainann=zeros(size(Ytrain,1),max(double(Ytrain)));
    for i=1:size(Ytrain,1)
        Ytrainann(i,Ytrain(i))=1;
    end
    ANN=[];
    for j=2:10
        disp(strcat('ANN',num2str(j)))
        waitbar(nalgo/numalgo)
        % Choose a Training Function
        % For a list of all training functions type: help nntrain
        % 'trainlm' is usually fastest.
        % 'trainbr' takes longer but may be better for challenging problems.
        % 'trainscg' uses less memory. Suitable in low memory situations.
        trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
        % Create a Pattern Recognition Network
        hiddenLayerSize = j;
        net = patternnet(hiddenLayerSize);
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
        tic
        [net] = train(net,Xtrain',Ytrainann');
        t=toc;
        Mdl.ANN{j}=net;
        Res.ANN{j}.time=t;
        
        % Test the Network
        outputs = net(Xtrain');
        [~,label_train] = max(outputs);
        C = confusionmat(Ytrain,categorical(label_train));
        
        Res.ANN{j}.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
        Res.ANN{j}.Train.Recall_T=C(1,1)/sum(C(1,:));
        Res.ANN{j}.Train.Selectivity_T=C(2,2)/sum(C(2,:));
        Res.ANN{j}.Train.Precision_T=C(1,1)/sum(C(:,1));
        Res.ANN{j}.Train.Fmeasure_T=2*Res.ANN{j}.Train.Precision_T*Res.ANN{j}.Train.Recall_T/(Res.ANN{j}.Train.Recall_T+Res.ANN{j}.Train.Precision_T);
        
        outputs = net(Xtest');
        [~,label_test] = max(outputs);
        C = confusionmat(Ytest,categorical(label_test));
        
        Res.ANN{j}.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
        Res.ANN{j}.Test.Recall_T=C(1,1)/sum(C(1,:));
        Res.ANN{j}.Test.Selectivity_T=C(2,2)/sum(C(2,:));
        Res.ANN{j}.Test.Precision_T=C(1,1)/sum(C(:,1));
        Res.ANN{j}.Test.Fmeasure_T=2*Res.ANN{j}.Test.Precision_T*Res.ANN{j}.Test.Recall_T/(Res.ANN{j}.Test.Recall_T+Res.ANN{j}.Test.Precision_T);
        
        ANN=[ANN; Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
        
        if j==2
            ANN2=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
        else if j==3
                ANN3=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
            else if j==4
                    ANN4=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
                else if j==5
                        ANN5=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
                    else if j==6
                            ANN6=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
                        else if j==7
                                ANN7=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
                            else if j==8
                                    ANN8=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
                                else if j==9
                                        ANN9=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
                                    else if j==10
                                            ANN10=[Res.ANN{j}.Train.Accuracy_T, Res.ANN{j}.Train.Recall_T, Res.ANN{j}.Train.Selectivity_T, Res.ANN{j}.Train.Precision_T,Res.ANN{j}.Train.Fmeasure_T,0,Res.ANN{j}.Test.Accuracy_T, Res.ANN{j}.Test.Recall_T, Res.ANN{j}.Test.Selectivity_T, Res.ANN{j}.Test.Precision_T,Res.ANN{j}.Test.Fmeasure_T];
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        C = confusionmat(Ytrain,categorical(label_train));
        Res.ANN{j}.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
        Res.ANN{j}.Train.Recall(1)=C(1,1)/sum(C(1,:));
        Res.ANN{j}.Train.Selectivity(1)=C(2,2)/sum(C(2,:));
        Res.ANN{j}.Train.Precision(1)=C(1,1)/sum(C(:,1));
        Res.ANN{j}.Train.Fmeasure(1)=2*Res.ANN{j}.Train.Precision(1)*Res.ANN{j}.Train.Recall(1)/(Res.ANN{j}.Train.Recall(1)+Res.ANN{j}.Train.Precision(1));
        
        C = confusionmat(Ytest,categorical(label_test));
        Res.ANN{j}.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
        Res.ANN{j}.Test.Recall(1)=C(1,1)/sum(C(1,:));
        Res.ANN{j}.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
        Res.ANN{j}.Test.Precision(1)=C(1,1)/sum(C(:,1));
        Res.ANN{j}.Test.Fmeasure(1)=2*Res.ANN{j}.Test.Precision(1)*Res.ANN{j}.Test.Recall(1)/(Res.ANN{j}.Test.Recall(1)+Res.ANN{j}.Test.Precision(1));
        
        nalgo=nalgo+1;
    end
end
if algo==0|sum(algo==4)==1
    %% CNN
    disp('CNN')
    waitbar(nalgo/numalgo)
    % Parameters
    epoch=50;
    minibatch=round(epoch/300*size(Ytrain,1));
    if minibatch<5
        minibatch=5;
    end
    if nargin<5
        poolnum=5;
    end
    
    Xtestc=reshape(Xtest',size(Xtest,2),1,1,size(Xtest,1));
    Xtrainc=reshape(Xtrain',size(Xtrain,2),1,1,size(Xtrain,1));
    poolnum=5;
    
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
        'ValidationData',{Xtestc,Ytest'},...
        'ExecutionEnvironment','parallel');
    
    layers = [
        imageInputLayer([size(Xtestc,1) size(Xtestc,2) size(Xtestc,3)])
        
        % Conv1
        convolution2dLayer([3 1],20,...
        'Stride',[1 1],'Padding',[1 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu1
        reluLayer
        
        % ConvPool2
        convolution2dLayer([3 1],poolnum,...
        'Stride',[2 1],'Padding',[1 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu2
        reluLayer
        
        % Conv3
        convolution2dLayer([3 3],35,...
        'Stride',[1 1],'Padding',[1 1],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu3
        reluLayer
        
        % ConvPool4
        convolution2dLayer([2 1],poolnum,...
        'Stride',[2 1],'Padding',[1 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu4
        reluLayer
        
        % Conv5
        convolution2dLayer([3 1],35,...
        'Stride',[1 1],'Padding',[1 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu5
        reluLayer
        
        % ConvPool6
        convolution2dLayer([1 1],poolnum,...
        'Stride',[2 1],'Padding',[1 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu6
        reluLayer
        
        % Conv7
        convolution2dLayer([3 1],35,...
        'Stride',[1 1],'Padding',[1 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu7
        reluLayer
        
        % ConvPool8
        convolution2dLayer([1 1],poolnum*2,...
        'Stride',[2 2],'Padding',[0 0],...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        
        % Relu8
        reluLayer
        
        fullyConnectedLayer(2,...
        'WeightLearnRateFactor',1,'WeightL2Factor',1,...
        'BiasLearnRateFactor',2,'BiasL2Factor',0)
        softmaxLayer
        classificationLayer];
    
    tic
    [net,info] = trainNetwork(Xtrainc,Ytrain',layers,options);
    Mdl.CNN=net;
    t=toc;
    Res.CNN.time=t;
    
    % Test the Network
    label_train = classify(net,Xtrainc);
    C = confusionmat(Ytrain,categorical(label_train));
    
    Res.CNN.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.CNN.Train.Recall_T=C(1,1)/sum(C(1,:));
    Res.CNN.Train.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.CNN.Train.Precision_T=C(1,1)/sum(C(:,1));
    Res.CNN.Train.Fmeasure_T=2*Res.CNN.Train.Precision_T*Res.CNN.Train.Recall_T/(Res.CNN.Train.Recall_T+Res.CNN.Train.Precision_T);
    
    label_test = classify(net,Xtestc);
    C = confusionmat(Ytest,categorical(label_test));
    
    Res.CNN.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.CNN.Test.Recall_T=C(1,1)/sum(C(1,:));
    Res.CNN.Test.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.CNN.Test.Precision_T=C(1,1)/sum(C(:,1));
    Res.CNN.Test.Fmeasure_T=2*Res.CNN.Test.Precision_T*Res.CNN.Test.Recall_T/(Res.CNN.Test.Recall_T+Res.CNN.Test.Precision_T);
    
    CNN=[Res.CNN.Train.Accuracy_T, Res.CNN.Train.Recall_T, Res.CNN.Train.Selectivity_T, Res.CNN.Train.Precision_T,Res.CNN.Train.Fmeasure_T,0,Res.CNN.Test.Accuracy_T, Res.CNN.Test.Recall_T, Res.CNN.Test.Selectivity_T, Res.CNN.Test.Precision_T,Res.CNN.Test.Fmeasure_T];
    
    C = confusionmat(Ytrain,categorical(label_train));
    Res.CNN.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.CNN.Train.Recall(1)=C(1,1)/sum(C(1,:));
    Res.CNN.Train.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.CNN.Train.Precision(1)=C(1,1)/sum(C(:,1));
    Res.CNN.Train.Fmeasure(1)=2*Res.CNN.Train.Precision(1)*Res.CNN.Train.Recall(1)/(Res.CNN.Train.Recall(1)+Res.CNN.Train.Precision(1));
    
    C = confusionmat(Ytest,categorical(label_test));
    Res.CNN.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.CNN.Test.Recall(1)=C(1,1)/sum(C(1,:));
    Res.CNN.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.CNN.Test.Precision(1)=C(1,1)/sum(C(:,1));
    Res.CNN.Test.Fmeasure(1)=2*Res.CNN.Test.Precision(1)*Res.CNN.Test.Recall(1)/(Res.CNN.Test.Recall(1)+Res.CNN.Test.Precision(1));
    
end
if algo==0|sum(algo==5)==1
    %% LDA
    disp('LDA')
    waitbar(nalgo/numalgo)
    tic
    Mdl.LDA = fitcdiscr(Xtrain,Ytrain);
    t=toc;
    Res.LDA.time=t;
    
    label_train = resubPredict(Mdl.LDA);
    C = confusionmat(Ytrain,label_train);
    
    Res.LDA.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.LDA.Train.Recall_T=C(1,1)/sum(C(1,:));
    Res.LDA.Train.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.LDA.Train.Precision_T=C(1,1)/sum(C(:,1));
    Res.LDA.Train.Fmeasure_T=2*Res.LDA.Train.Precision_T*Res.LDA.Train.Recall_T/(Res.LDA.Train.Recall_T+Res.LDA.Train.Precision_T);
    
    [label_test,score_test,cost_test]  = predict(Mdl.LDA,Xtest);
    C = confusionmat(Ytest,label_test);
    
    Res.LDA.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.LDA.Test.Recall_T=C(1,1)/sum(C(1,:));
    Res.LDA.Test.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.LDA.Test.Precision_T=C(1,1)/sum(C(:,1));
    Res.LDA.Test.Fmeasure_T=2*Res.LDA.Test.Precision_T*Res.LDA.Test.Recall_T/(Res.LDA.Test.Recall_T+Res.LDA.Test.Precision_T);
    
    LDA=[Res.LDA.Train.Accuracy_T, Res.LDA.Train.Recall_T, Res.LDA.Train.Selectivity_T, Res.LDA.Train.Precision_T,Res.LDA.Train.Fmeasure_T,0,Res.LDA.Test.Accuracy_T, Res.LDA.Test.Recall_T, Res.LDA.Test.Selectivity_T, Res.LDA.Test.Precision_T,Res.LDA.Test.Fmeasure_T];
    
    C = confusionmat(Ytrain,label_train);
    Res.LDA.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.LDA.Train.Recall(1)=C(1,1)/sum(C(1,:));
    Res.LDA.Train.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.LDA.Train.Precision(1)=C(1,1)/sum(C(:,1));
    Res.LDA.Train.Fmeasure(1)=2*Res.LDA.Train.Precision(1)*Res.LDA.Train.Recall(1)/(Res.LDA.Train.Recall(1)+Res.LDA.Train.Precision(1));
    
    C = confusionmat(Ytest,label_test);
    Res.LDA.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.LDA.Test.Recall(1)=C(1,1)/sum(C(1,:));
    Res.LDA.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.LDA.Test.Precision(1)=C(1,1)/sum(C(:,1));
    Res.LDA.Test.Fmeasure(1)=2*Res.LDA.Test.Precision(1)*Res.LDA.Test.Recall(1)/(Res.LDA.Test.Recall(1)+Res.LDA.Test.Precision(1));
    
    nalgo=nalgo+1;
end
if algo==0|sum(algo==6)==1
    %% QDA
    disp('QDA')
    waitbar(nalgo/numalgo)
    tic
    Mdl.QDA = fitcdiscr(Xtrain,Ytrain,'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('Holdout',0.3,...
        'AcquisitionFunctionName','expected-improvement-plus'));
    t=toc;
    Res.QDA.time=t;
    
    label_train = resubPredict(Mdl.QDA);
    C = confusionmat(Ytrain,label_train);
    
    Res.QDA.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.QDA.Train.Recall_T=C(1,1)/sum(C(1,:));
    Res.QDA.Train.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.QDA.Train.Precision_T=C(1,1)/sum(C(:,1));
    Res.QDA.Train.Fmeasure_T=2*Res.QDA.Train.Precision_T*Res.QDA.Train.Recall_T/(Res.QDA.Train.Recall_T+Res.QDA.Train.Precision_T);
    
    [label_test,score_test,cost_test]  = predict(Mdl.QDA,Xtest);
    C = confusionmat(Ytest,label_test);
    
    Res.QDA.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.QDA.Test.Recall_T=C(1,1)/sum(C(1,:));
    Res.QDA.Test.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.QDA.Test.Precision_T=C(1,1)/sum(C(:,1));
    Res.QDA.Test.Fmeasure_T=2*Res.QDA.Test.Precision_T*Res.QDA.Test.Recall_T/(Res.QDA.Test.Recall_T+Res.QDA.Test.Precision_T);
    
    QDA=[Res.QDA.Train.Accuracy_T, Res.QDA.Train.Recall_T, Res.QDA.Train.Selectivity_T, Res.QDA.Train.Precision_T,Res.QDA.Train.Fmeasure_T,0,Res.QDA.Test.Accuracy_T, Res.QDA.Test.Recall_T, Res.QDA.Test.Selectivity_T, Res.QDA.Test.Precision_T,Res.QDA.Test.Fmeasure_T];
    
    C = confusionmat(Ytrain,label_train);
    Res.QDA.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.QDA.Train.Recall(1)=C(1,1)/sum(C(1,:));
    Res.QDA.Train.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.QDA.Train.Precision(1)=C(1,1)/sum(C(:,1));
    Res.QDA.Train.Fmeasure(1)=2*Res.QDA.Train.Precision(1)*Res.QDA.Train.Recall(1)/(Res.QDA.Train.Recall(1)+Res.QDA.Train.Precision(1));
    
    C = confusionmat(Ytest,label_test);
    Res.QDA.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.QDA.Test.Recall(1)=C(1,1)/sum(C(1,:));
    Res.QDA.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.QDA.Test.Precision(1)=C(1,1)/sum(C(:,1));
    Res.QDA.Test.Fmeasure(1)=2*Res.QDA.Test.Precision(1)*Res.QDA.Test.Recall(1)/(Res.QDA.Test.Recall(1)+Res.QDA.Test.Precision(1));
    
    nalgo=nalgo+1;
end
if algo==0|sum(algo==7)==1
    %% SVM
    disp('SVM')
    waitbar(nalgo/numalgo)
    tic
    Mdl.SVM = fitcsvm(Xtrain,Ytrain);
    t=toc;
    Res.SVM.time=t;
    
    label_train = resubPredict(Mdl.SVM);
    C = confusionmat(Ytrain,label_train);
    
    Res.SVM.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.SVM.Train.Recall_T=C(1,1)/sum(C(1,:));
    Res.SVM.Train.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.SVM.Train.Precision_T=C(1,1)/sum(C(:,1));
    Res.SVM.Train.Fmeasure_T=2*Res.SVM.Train.Precision_T*Res.SVM.Train.Recall_T/(Res.SVM.Train.Recall_T+Res.SVM.Train.Precision_T);
    
    [label_test,score_test]  = predict(Mdl.SVM,Xtest);
    C = confusionmat(Ytest,label_test);
    
    Res.SVM.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.SVM.Test.Recall_T=C(1,1)/sum(C(1,:));
    Res.SVM.Test.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.SVM.Test.Precision_T=C(1,1)/sum(C(:,1));
    Res.SVM.Test.Fmeasure_T=2*Res.SVM.Test.Precision_T*Res.SVM.Test.Recall_T/(Res.SVM.Test.Recall_T+Res.SVM.Test.Precision_T);
    
    SVM=[Res.SVM.Train.Accuracy_T, Res.SVM.Train.Recall_T, Res.SVM.Train.Selectivity_T, Res.SVM.Train.Precision_T,Res.SVM.Train.Fmeasure_T,0,Res.SVM.Test.Accuracy_T, Res.SVM.Test.Recall_T, Res.SVM.Test.Selectivity_T, Res.SVM.Test.Precision_T,Res.SVM.Test.Fmeasure_T];
    
    C = confusionmat(Ytrain,label_train);
    Res.SVM.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.SVM.Train.Recall(1)=C(1,1)/sum(C(1,:));
    Res.SVM.Train.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.SVM.Train.Precision(1)=C(1,1)/sum(C(:,1));
    Res.SVM.Train.Fmeasure(1)=2*Res.SVM.Train.Precision(1)*Res.SVM.Train.Recall(1)/(Res.SVM.Train.Recall(1)+Res.SVM.Train.Precision(1));
    
    C = confusionmat(Ytest,label_test);
    Res.SVM.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.SVM.Test.Recall(1)=C(1,1)/sum(C(1,:));
    Res.SVM.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.SVM.Test.Precision(1)=C(1,1)/sum(C(:,1));
    Res.SVM.Test.Fmeasure(1)=2*Res.SVM.Test.Precision(1)*Res.SVM.Test.Recall(1)/(Res.SVM.Test.Recall(1)+Res.SVM.Test.Precision(1));
    
    nalgo=nalgo+1;
end
if algo==0|sum(algo==8)==1
    %% PLSDA
    disp('PLSDA')
    waitbar(nalgo/numalgo)
    tic
    % Centering of the spectra
    Sc_cal=Centrer(double(Xtrain));
    Sc_val=Centrerval(double(Xtest),double(Xtrain));
    Mdl.PLSDA.Sc_cal=double(Xtrain);
    % Definition of the Y PLS-DA
    Y_cal = YPLSDA(length(unique(double(Ytrain))),double(Ytrain));
    Y_val = YPLSDA(length(unique(double(Ytest))),double(Ytest));
    Mdl.PLSDA.Y_cal=Y_cal;
    % Saving the references values
    Y_calb=Y_cal;
    Y_valb=Y_val;
    
    % Centering of the calibration references
    Y_cal=Centrer(Y_cal);
    
    % Model creation
    [ Mdl.PLSDA.B,Mdl.PLSDA.Wstar,Mdl.PLSDA.T,Mdl.PLSDA.P,Mdl.PLSDA.Q,Mdl.PLSDA.W,Mdl.PLSDA.R2X,Mdl.PLSDA.R2Y]=plsnipals(double(Sc_cal),Y_cal);
    Mdl.PLSDA.R2X=1-Mdl.PLSDA.R2X;
    Mdl.PLSDA.R2Y=1-Mdl.PLSDA.R2Y;
    
    t=toc;
    Res.PLSDA.time=t;
    
    Y0 = Sc_cal*Mdl.PLSDA.Wstar*Mdl.PLSDA.Q; % Calibration prediction
    label_train = Decentrerval( Y0, Y_calb ); % Uncentering of the prediction
    [~,label_train] = max(label_train');
    C = confusionmat(double(Ytrain),label_train');
    
    Res.PLSDA.Train.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.PLSDA.Train.Recall_T=C(1,1)/sum(C(1,:));
    Res.PLSDA.Train.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.PLSDA.Train.Precision_T=C(1,1)/sum(C(:,1));
    Res.PLSDA.Train.Fmeasure_T=2*Res.PLSDA.Train.Precision_T*Res.PLSDA.Train.Recall_T/(Res.PLSDA.Train.Recall_T+Res.PLSDA.Train.Precision_T);
    
    Y0 = Sc_val*Mdl.PLSDA.Wstar*Mdl.PLSDA.Q; % Calibration prediction
    label_test = Decentrerval( Y0, Y_calb ); % Uncentering of the prediction
    [~,label_test] = max(label_test');
    C = confusionmat(double(Ytest),label_test');
    
    Res.PLSDA.Test.Accuracy_T=sum(diag(C))/sum(C(:)); % Accuracy
    Res.PLSDA.Test.Recall_T=C(1,1)/sum(C(1,:));
    Res.PLSDA.Test.Selectivity_T=C(2,2)/sum(C(2,:));
    Res.PLSDA.Test.Precision_T=C(1,1)/sum(C(:,1));
    Res.PLSDA.Test.Fmeasure_T=2*Res.PLSDA.Test.Precision_T*Res.PLSDA.Test.Recall_T/(Res.PLSDA.Test.Recall_T+Res.PLSDA.Test.Precision_T);
    
    PLSDA=[Res.PLSDA.Train.Accuracy_T, Res.PLSDA.Train.Recall_T, Res.PLSDA.Train.Selectivity_T, Res.PLSDA.Train.Precision_T,Res.PLSDA.Train.Fmeasure_T,0,Res.PLSDA.Test.Accuracy_T, Res.PLSDA.Test.Recall_T, Res.PLSDA.Test.Selectivity_T, Res.PLSDA.Test.Precision_T,Res.PLSDA.Test.Fmeasure_T];
    
    C = confusionmat(double(Ytrain),label_train);
    Res.PLSDA.Train.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.PLSDA.Train.Recall(1)=C(1,1)/sum(C(1,:));
    Res.PLSDA.Train.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.PLSDA.Train.Precision(1)=C(1,1)/sum(C(:,1));
    Res.PLSDA.Train.Fmeasure(1)=2*Res.PLSDA.Train.Precision(1)*Res.PLSDA.Train.Recall(1)/(Res.PLSDA.Train.Recall(1)+Res.PLSDA.Train.Precision(1));
    
    C = confusionmat(double(Ytest),label_test);
    Res.PLSDA.Test.Accuracy(1)=sum(diag(C))/sum(C(:)); % Accuracy
    Res.PLSDA.Test.Recall(1)=C(1,1)/sum(C(1,:));
    Res.PLSDA.Test.Selectivity(1)=C(2,2)/sum(C(2,:));
    Res.PLSDA.Test.Precision(1)=C(1,1)/sum(C(:,1));
    Res.PLSDA.Test.Fmeasure(1)=2*Res.PLSDA.Test.Precision(1)*Res.PLSDA.Test.Recall(1)/(Res.PLSDA.Test.Recall(1)+Res.PLSDA.Test.Precision(1));
    
    nalgo=nalgo+1;
end
close(h)

% Display discriminant wavelength for each model
figure;
i=1;
% if sum(algo==1)>0||algo==0
% plot(wl,model.DT.NodeProbability,'linewidth',2)
% hold on
% leg{i}='DT';
% i=i+1;
% end
if sum(algo==3)>0||(length(algo)==1&&algo==0)
    plot(wl,sum(cell2mat(Mdl.ANN{1,2}.IW))/max(abs(sum(cell2mat(Mdl.ANN{1,2}.IW)))),'linewidth',2)
    hold on
    leg{i}='ANN2';
    i=i+1;
end
if sum(algo==5)>0||(length(algo)==1&&algo==0)
    plot(wl,Mdl.LDA.Coeffs(1, 2).Linear/max(abs(Mdl.LDA.Coeffs(1, 2).Linear)),'linewidth',2)
    hold on
    leg{i}='LDA';
    i=i+1;
end
if sum(algo==6)>0||(length(algo)==1&&algo==0)
    plot(wl,Mdl.QDA.Coeffs(1, 2).Linear/max(abs(Mdl.QDA.Coeffs(1, 2).Linear)),'linewidth',2)
    hold on
    leg{i}='QDA';
    i=i+1;
end
if sum(algo==7)>0||(length(algo)==1&&algo==0)
    plot(wl,Mdl.SVM.Beta/max(abs(Mdl.SVM.Beta)),'linewidth',2)
    hold on
    leg{i}='SVM';
    i=i+1;
end
if sum(algo==8)>0||(length(algo)==1&&algo==0)
    plot(wl,Mdl.PLSDA.B(:,1)/max(abs(Mdl.PLSDA.B(:,1))),'linewidth',2)
    leg{i}='PLS-DA';
end
grid on
xlim([wl(1) wl(end)])
legend(leg)
xlabel('Wavelength (nm)')
ylabel('Normalized coefficients')
set(gca,'fontsize',14,'xtick',round(wl(1)/100)*100:100:round(wl(end)/100)*100)

% Remove abberant pixels
mask=AbberantPixels(M,RGB,d,wl);

if sum(algo==3)==1|algo==0
    % ANN opt
    annres=mean([ANN(:,1) ANN(:,7)],2)./(1+abs(ANN(:,1)-ANN(:,7))); % Accuracies normalized by difference accuracies
    [~,idx]=max(annres);
end

if fig==1
    % Map
    figure
    if (length(algo)==1&&algo==0)
        ha(1)=subplot(511);
        imagesc(double(reshape(predict(Mdl.DT,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
        title('Decision tree')
        ha(2)=subplot(512);
        imagesc(double(reshape(predict(Mdl.RF,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
        title('Random forest')
        ha(3)=subplot(513);
        net=Mdl.ANN{idx(1)};
        output=net(reshape(M,[],size(M,3))');
        [~,label] = max(output);
        imagesc(reshape(label,size(M,1),size(M,2)).*mask);
        title('Artificial Neural Network')
        ha(4)=subplot(514);
        net=Mdl.CNN;
        label=classify(net,reshape(permute(M,[3 1 2]),size(M,3),1,1,size(M,2)*size(M,1)).*mask);
        imagesc(reshape(label,size(M,1),size(M,2)));
        title('Convolutional Neural Network')
        ha(5)=subplot(515);
        imagesc(RGB);
        title('RGB')
    else if length(algo)==1
            if sum(algo==1)==1
                ha(1)=subplot(211);
                imagesc(double(reshape(predict(Mdl.DT,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
                title('Decision tree')
                ha(2)=subplot(212);
                imagesc(RGB);
                title('RGB')
            else if sum(algo==2)==1
                    ha(1)=subplot(211);
                    imagesc(double(reshape(predict(Mdl.RF,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
                    title('Random forest')
                    ha(2)=subplot(212);
                    imagesc(RGB);
                    title('RGB')
                else if sum(algo==3)==1
                        ha(1)=subplot(211);
                        net=Mdl.ANN{idx(1)};
                        output=net(reshape(M,[],size(M,3))');
                        [~,label] = max(output);
                        imagesc(reshape(label,size(M,1),size(M,2)).*mask);
                        title('Artificial Neural Network')
                        ha(2)=subplot(212);
                        imagesc(RGB);
                        title('RGB')
                    else
                        ha(1)=subplot(211);
                        net=Mdl.CNN;
                        label=classify(net,reshape(permute(M,[3 1 2]),size(M,3),1,1,size(M,2)*size(M,1)));
                        imagesc(reshape(label,size(M,1),size(M,2)).*mask);
                        title('Convolutional Neural Network')
                        ha(2)=subplot(212);
                        imagesc(RGB);
                        title('RGB')
                    end
                end
            end
        else
            it=1;
            for j=1:length(algo)
                if algo(j)==1
                    ha(it)=subplot(length(algo)+1,1,it);
                    imagesc(d,d(1:size(M,3)),double(reshape(predict(Mdl.DT,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
                    title('Decision tree')
                    
                    IM(:,:,it)=double(reshape(predict(Mdl.DT,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask;
                    IMlabel{it}='Decision tree';
                    for k=1:length(Y)
                        Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                    end
                    Mdl.Perf.DT_rvs=corr(Y,Ys');
                    Mdl.Perf.DT_Ss=median(nanstd(squeeze(IM(:,:,it))));
                    DT=[DT 0 Mdl.Perf.DT_rvs Mdl.Perf.DT_Ss];
                    lab{it}='DT';
                    it=it+1;
                else if algo(j)==2
                        ha(it)=subplot(length(algo)+1,1,it);
                        imagesc(d,d(1:size(M,3)),double(reshape(predict(Mdl.RF,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
                        title('Random forest')
                        
                        IM(:,:,it)=double(reshape(predict(Mdl.RF,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask;
                        for k=1:length(Y)
                            Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                        end
                        Mdl.Perf.RF_rvs=corr(Y,Ys');
                        Mdl.Perf.RF_Ss=median(nanstd(squeeze(IM(:,:,it))));
                        RF=[RF 0 Mdl.Perf.RF_rvs Mdl.Perf.RF_Ss];
                        IMlabel{it}='Random forest';
                        lab{it}='RF';
                        it=it+1;
                    else if algo(j)==3
                            ha(it)=subplot(length(algo)+1,1,it);
                            net=Mdl.ANN{idx(1)};
                            output=net(reshape(M,size(M,1)*size(M,2),size(M,3))');
                            [~,label] = max(output);
                            imagesc(d,d(1:size(M,3)),reshape(label,size(M,1),size(M,2)).*mask);
                            title('Artificial Neural Network')
                            
                            IM(:,:,it)=reshape(label,size(M,1),size(M,2)).*mask;
                            for k=1:length(Y)
                                Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                            end
                            Mdl.Perf.ANN_rvs=corr(Y,Ys');
                            Mdl.Perf.ANN_Ss=median(nanstd(squeeze(IM(:,:,it))));
                            ANNtmp=zeros(9,3);
                            ANNtmp(idx(1),2)=Mdl.Perf.ANN_rvs;
                            ANNtmp(idx(1),3)=Mdl.Perf.ANN_Ss;
                            ANN=[ANN ANNtmp];
                            if idx(1)==2;ANN2=[ANN2 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                ANN3=[ANN3 0 0 0];ANN4=[ANN4 0 0 0];ANN5=[ANN5 0 0 0];
                                ANN6=[ANN6 0 0 0];ANN7=[ANN7 0 0 0];ANN8=[ANN8 0 0 0];
                                ANN9=[ANN9 0 0 0];ANN10=[ANN10 0 0 0];
                            else if idx(1)==3;ANN3=[ANN3 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                    ANN2=[ANN2 0 0 0];ANN4=[ANN4 0 0 0];ANN5=[ANN5 0 0 0];
                                    ANN6=[ANN6 0 0 0];ANN7=[ANN7 0 0 0];ANN8=[ANN8 0 0 0];
                                    ANN9=[ANN9 0 0 0];ANN10=[ANN10 0 0 0];
                                else if idx(1)==4;ANN4=[ANN4 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                        ANN3=[ANN3 0 0 0];ANN2=[ANN2 0 0 0];ANN5=[ANN5 0 0 0];
                                        ANN6=[ANN6 0 0 0];ANN7=[ANN7 0 0 0];ANN8=[ANN8 0 0 0];
                                        ANN9=[ANN9 0 0 0];ANN10=[ANN10 0 0 0];
                                    else if idx(1)==5;ANN5=[ANN5 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                            ANN3=[ANN3 0 0 0];ANN4=[ANN4 0 0 0];ANN2=[ANN2 0 0 0];
                                            ANN6=[ANN6 0 0 0];ANN7=[ANN7 0 0 0];ANN8=[ANN8 0 0 0];
                                            ANN9=[ANN9 0 0 0];ANN10=[ANN10 0 0 0];
                                        else if idx(1)==6;ANN6=[ANN6 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                                ANN3=[ANN3 0 0 0];ANN4=[ANN4 0 0 0];ANN5=[ANN5 0 0 0];
                                                ANN2=[ANN2 0 0 0];ANN7=[ANN7 0 0 0];ANN8=[ANN8 0 0 0];
                                                ANN9=[ANN9 0 0 0];ANN10=[ANN10 0 0 0];
                                            else if idx(1)==7;ANN7=[ANN7 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                                    ANN3=[ANN3 0 0 0];ANN4=[ANN4 0 0 0];ANN5=[ANN5 0 0 0];
                                                    ANN6=[ANN6 0 0 0];ANN2=[ANN2 0 0 0];ANN8=[ANN8 0 0 0];
                                                    ANN9=[ANN9 0 0 0];ANN10=[ANN10 0 0 0];
                                                else if idx(1)==8;ANN8=[ANN8 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                                        ANN3=[ANN3 0 0 0];ANN4=[ANN4 0 0 0];ANN5=[ANN5 0 0 0];
                                                        ANN6=[ANN6 0 0 0];ANN7=[ANN7 0 0 0];ANN2=[ANN2 0 0 0];
                                                        ANN9=[ANN9 0 0 0];ANN10=[ANN10 0 0 0];
                                                    else if idx(1)==9;ANN9=[ANN9 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                                            ANN3=[ANN3 0 0 0];ANN4=[ANN4 0 0 0];ANN5=[ANN5 0 0 0];
                                                            ANN6=[ANN6 0 0 0];ANN7=[ANN7 0 0 0];ANN8=[ANN8 0 0 0];
                                                            ANN2=[ANN2 0 0 0];ANN10=[ANN10 0 0 0];
                                                        else if idx(1)==10;ANN10=[ANN10 0 Mdl.Perf.ANN_rvs Mdl.Perf.ANN_Ss];
                                                                ANN3=[ANN3 0 0 0];ANN4=[ANN4 0 0 0];ANN5=[ANN5 0 0 0];
                                                                ANN6=[ANN6 0 0 0];ANN7=[ANN7 0 0 0];ANN8=[ANN8 0 0 0];
                                                                ANN9=[ANN9 0 0 0];ANN2=[ANN2 0 0 0];
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                            IMlabel{it}='Artificial Neural Network';
                            lab{it}='ANN';
                            it=it+1;
                        else if algo(j)==4
                                ha(it)=subplot(length(algo)+1,1,it);
                                net=Mdl.CNN;
                                label=classify(net,reshape(permute(M,[3 1 2]),size(M,3),1,1,size(M,2)*size(M,1)));
                                imagesc(d,d(1:size(M,3)),reshape(label,size(M,1),size(M,2)).*mask);
                                title('Convolutional Neural Network')
                                
                                IM(:,:,it)=reshape(label,size(M,1),size(M,2)).*mask;
                                for k=1:length(Y)
                                    Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                                end
                                Mdl.Perf.CNN_rvs=corr(Y,Ys');
                                Mdl.Perf.CNN_Ss=median(nanstd(squeeze(IM(:,:,it))));
                                CNN=[CNN 0 Mdl.Perf.CNN_rvs Mdl.Perf.CNN_Ss];
                                IMlabel{it}='Convolutional Neural Network';
                                lab{it}='CNN';
                                it=it+1;
                            else if algo(j)==5
                                    ha(it)=subplot(length(algo)+1,1,it);
                                    imagesc(d,d(1:size(M,3)),double(reshape(predict(Mdl.LDA,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
                                    title('Linear discriminant analysis')
                                    
                                    IM(:,:,it)=double(reshape(predict(Mdl.LDA,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask;
                                    for k=1:length(Y)
                                        Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                                    end
                                    Mdl.Perf.LDA_rvs=corr(Y,Ys');
                                    Mdl.Perf.LDA_Ss=median(nanstd(squeeze(IM(:,:,it))));
                                    LDA=[LDA 0 Mdl.Perf.LDA_rvs Mdl.Perf.LDA_Ss];
                                    IMlabel{it}='Linear discriminant analysis';
                                    lab{it}='LDA';
                                    it=it+1;
                                else if algo(j)==6
                                        ha(it)=subplot(length(algo)+1,1,it);
                                        imagesc(d,d(1:size(M,3)),double(reshape(predict(Mdl.QDA,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
                                        title('Quadratic discriminant analysis')
                                        
                                        IM(:,:,it)=double(reshape(predict(Mdl.QDA,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask;
                                        for k=1:length(Y)
                                            Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                                        end
                                        Mdl.Perf.QDA_rvs=corr(Y,Ys');
                                        Mdl.Perf.QDA_Ss=median(nanstd(squeeze(IM(:,:,it))));
                                        QDA=[QDA 0 Mdl.Perf.QDA_rvs Mdl.Perf.QDA_Ss];
                                        IMlabel{it}='Quadratic discriminant analysis';
                                        lab{it}='QDA';
                                        it=it+1;
                                    else if algo(j)==7
                                            ha(it)=subplot(length(algo)+1,1,it);
                                            imagesc(d,d(1:size(M,3)),double(reshape(predict(Mdl.SVM,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask);
                                            title('Support vector machine')
                                            
                                            IM(:,:,it)=double(reshape(predict(Mdl.SVM,reshape(M,[],size(M,3))),size(M,1),size(M,2))).*mask;
                                            for k=1:length(Y)
                                                Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                                            end
                                            Mdl.Perf.SVM_rvs=corr(Y,Ys');
                                            Mdl.Perf.SVM_Ss=median(nanstd(squeeze(IM(:,:,it))));
                                            SVM=[SVM 0 Mdl.Perf.SVM_rvs Mdl.Perf.SVM_Ss];
                                            IMlabel{it}='Support vector machine';
                                            lab{it}='SVM';
                                            it=it+1;
                                        else if algo(j)==8
                                                ha(it)=subplot(length(algo)+1,1,it);
                                                Sc_cal=Centrer(double(reshape(M,[],size(M,3))));
                                                Y0 = Sc_cal*Mdl.PLSDA.Wstar*Mdl.PLSDA.Q; % Calibration prediction
                                                label = Decentrerval( Y0, Y_calb ); % Uncentering of the prediction
                                                [~,label] = max(label');
                                                imagesc(d,d(1:size(M,3)),reshape(label,size(M,1),size(M,2)).*mask);
                                                title('Partial least squares discriminant analysis')
                                                
                                                IM(:,:,it)=reshape(label,size(M,1),size(M,2)).*mask;
                                                for k=1:length(Y)
                                                    Ys(k)=nanmedian(nanmedian(squeeze(IM(:,disc(1,k):disc(2,k),it))));
                                                end
                                                Mdl.Perf.PLSDA_rvs=corr(Y,Ys');
                                                Mdl.Perf.PLSDA_Ss=median(nanstd(squeeze(IM(:,:,it))));
                                                PLSDA=[PLSDA 0 Mdl.Perf.PLSDA_rvs Mdl.Perf.PLSDA_Ss];
                                                IMlabel{it}='Partial least squares discriminant analysis';
                                                lab{it}='PLS-DA';
                                                it=it+1;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            ha(it)=subplot(length(algo)+1,1,it);
            %                     imagesc(d,d(1:size(Mt,3)),RGB{i}(round(2/5*size(RGB{i},1)):round(3/5*size(RGB{i},1)),:,:).*(0.5/mean(mean(RGB{i}(round(2/5*size(RGB{i},1)):round(3/5*size(RGB{i},1)),:,:)))));
            imagesc(d,d(1:size(M,3)),IMref)
            colormap(ha(it),[0 0 0;parula(max(IMref(:)))])
            title('Reference')
        end
        linkaxes(ha,'xy')
    end
end

if exist('Y','var')
    ha=[];
    figure
    ha(1)=subplot(211);
    imagesc(d,d(1:size(RGB,1)),RGB)
    set(gca,'fontsize',14)
    xlim([d(1) d(end)])
    ha(2)=subplot(212);
    plot(d,squeeze(mode(IM,1))+repmat((1:size(IM,3))*1.5,size(IM,2),1),'linewidth',2)
    hold on
    for i=1:length(dY)
        if strcmp(dataref,'Max')
            plot([dY(i)-datadref(i) dY(i)],[Y(i) Y(i)],'g','linewidth',2)
            hold on
        else if strcmp(dataref,'Min')
                plot([dY(i) dY(i)+datadref(i)],[Y(i) Y(i)],'g','linewidth',2)
                hold on
            else
                plot([dY(i)-0.5*datadref(i) dY(i)+0.5*datadref(i)],[Y(i) Y(i)],'g','linewidth',2)
                hold on
            end
        end
    end
    grid on
    lab{it}='Ref';
    xlim([d(1) d(end)])
    legend(lab)
    set(gca,'fontsize',14)
    linkaxes(ha,'x')
end

if exist('Y','var')
    Parameters={'Train Accuracy','Train Recall','Train Selectivity','Train Precision','Train Fmeasure','','Test Accuracy','Test Recall','Test Selectivity','Test Precision','Test Fmeasure','','Rvs','Ss'}';
else
    Parameters={'Train Accuracy','Train Recall','Train Selectivity','Train Precision','Train Fmeasure','','Test Accuracy','Test Recall','Test Selectivity','Test Precision','Test Fmeasure'}';
end
if algo==0
    DT=DT';RF=RF';ANN=ANN';CNN=CNN';LDA=LDA';QDA=QDA';SVM=SVM';PLSDA=PLSDA';
    Recap=table(Parameters,DT,RF,ANN,CNN,LDA,QDA,SVM,PLSDA)
else if length(algo)==1
        if sum(algo==1)==1
            DT=DT';
            Recap=table(Parameters,DT)
        else if sum(algo==2)==1
                RF=RF';
                Recap=table(Parameters,RF)
            else if sum(algo==3)==1
                    ANN=ANN';
                    Recap=table(Parameters,ANN)
                else if sum(algo==4)==1
                        CNN=CNN';
                        Recap=table(Parameters,CNN)
                    else if sum(algo==5)==1
                            LDA=LDA';
                            Recap=table(Parameters,LDA)
                        else if sum(algo==6)==1
                                QDA=QDA';
                                Recap=table(Parameters,QDA)
                            else if sum(algo==7)==1
                                    SVM=SVM';
                                    Recap=table(Parameters,SVM)
                                else
                                    PLSDA=PLSDA';
                                    Recap=table(Parameters,PLSDA)
                                end
                            end
                        end
                    end
                end
            end
        end
    else
        Recap=table(Parameters);
        for i=1:length(algo)
            if algo(i)==1
                DT=DT';
                Recap=[Recap table(DT)];
            else if algo(i)==2
                    RF=RF';
                    Recap=[Recap table(RF)];
                else if algo(i)==3
                        ANN2=ANN2';ANN3=ANN3';ANN4=ANN4';ANN5=ANN5';
                        ANN6=ANN6';ANN7=ANN7';ANN8=ANN8';ANN9=ANN9';
                        ANN10=ANN10';
                        Recap=[Recap table(ANN2,ANN3,ANN4,ANN5,ANN6,ANN7,ANN8,ANN9,ANN10)];
                    else if algo(i)==4
                            CNN=CNN';
                            Recap=[Recap table(CNN)];
                        else if algo(i)==5
                                LDA=LDA';
                                Recap=[Recap table(LDA)];
                            else if algo(i)==6
                                    QDA=QDA';
                                    Recap=[Recap table(QDA)];
                                else if algo(i)==7
                                        SVM=SVM';
                                        Recap=[Recap table(SVM)];
                                    else
                                        PLSDA=PLSDA';
                                        Recap=[Recap table(PLSDA)];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        Recap
        Mdl.Recap=Recap;
    end
end
Mdl.Res=Res;

end