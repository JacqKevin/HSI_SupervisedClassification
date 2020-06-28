function ClassifMap=Discrimination_Apply(M,RGB,d,wl,model,indx,nn)
% Function to apply a classification model created with Discrimination_Create.
%
% INPUT:
%       M: HSI datacube (n*m*p)
%       RGB: Image of the sample
%       d : Associated depth (1*m)(mm)
%       wl: Associated wavelength (1*p)
%	model: Structures created in Discrimination_Create
%	indx: Number of the discrimination algorithm 
%		in the structure model to use. 
%	nn (if ANN approach selected): Number of neurons 
%		to use for the ANN approach
% OUTPUT:
%       ClassifMap: Predicted classification maps
%
% This is the Matalb toolbox from the papers:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Jacq, K., Rapuc, W., Benoit, A., Develle, A.-L., Coquin, D., 
% Fanget, B., Perrette, Y.,  Sabatier, P., Wilhelm, B., 
% Debret, M., Arnaud, F., 2019. Sedimentary structures 
% discriminations with hyperspectral imaging on 
% sediment cores. Computers & Geosciences

% Please cite our papers if you use our code for your research.


% Find the approaches used
iter=1;
if isfield(model,'DT')
    lbl{iter}='DT';
    iter=iter+1;
end
if isfield(model,'RF')
    lbl{iter}='RF';
    iter=iter+1;
end
if isfield(model,'ANN')
    lbl{iter}='ANN';
    iter=iter+1;
end
if isfield(model,'CNN')
    lbl{iter}='CNN';
    iter=iter+1;
end
if isfield(model,'LDA')
    lbl{iter}='LDA';
    iter=iter+1;
end
if isfield(model,'QDA')
    lbl{iter}='QDA';
    iter=iter+1;
end
if isfield(model,'PLSDA')
    lbl{iter}='PLSDA';
    iter=iter+1;
end

% Display to a user the available models
if nargin<6
    [indx,tf] = listdlg('PromptString','Which model to use:',...
        'SelectionMode','single',...
        'ListString',lbl);
end

% Parameters
if isfield(model,'idx_wl1')
    M=M(:,:,model.idx_wl1);
    wl=wl(model.idx_wl1);
end

% Conversion in absorbance if necessary
if strcmp(model.Spectral,'Absorbance')
    S=reshape(M,size(M,1)*size(M,2),size(M,3));
    [a,b]=find(S==0);
    S(a,b)=eps;
    S=log(1./S);
else
    S=reshape(M,size(M,1)*size(M,2),size(M,3));
end

% First preprocessing
if strcmp(model.SpectralProcessing,'Raw')==0
    pret={'Detrend','SNV','SNVD','MSC','D1','D2','SNV+D1','SNVD+D1','MSC+D1','SNV+D2','SNVD+D2','MSC+D2','CR'};
    [~,b]=find(double(strcmp(pret,model.SpectralProcessing))==1);
    S = AllPret(S,wl,b);
end

% Second preprocessing
if strcmp(model.Preprocessing,'Center')
    [S,~,~]=pretreat(S,'center',model.PreprocessingPara.para1,model.PreprocessingPara.para2);
else if strcmp(model.Preprocessing,'Autoscaling')
        [S,~,~]=pretreat(S,'autoscaling',model.PreprocessingPara.para1,model.PreprocessingPara.para2);
    end
end

% Reduced wavelengths
if isfield(model,'idx_wl2')
    idx_wl2=model.idx_wl2;
else
    idx_wl2=zeros(1,length(model.idx_wl1));
end

% Prediction
X=S(:,idx_wl2==0);
if strcmp(lbl{indx},'DT')
    [label] = predict(model.DT,X);
end
if strcmp(lbl{indx},'RF')
    [label] = predict(model.RF,X);
end
if strcmp(lbl{indx},'CNN')
    net=model.CNN;
    label=classify(net,reshape(permute(M,[3 1 2]),size(M,3),1,1,size(M,2)*size(M,1)));
end
if strcmp(lbl{indx},'ANN')
    if nargin<5
        [indx,tf] = listdlg('PromptString','How many neuron to use?',...
            'SelectionMode','single',...
            'ListString',num2str((1:length(model.ANN))'));
    else
        indx=nn;
    end
    net=model.ANN{indx};
    outputs = net(X');
    [~,label] = max(outputs);
end
if strcmp(lbl{indx},'LDA')
    [label] = predict(model.LDA,X);
end
if strcmp(lbl{indx},'QDA')
    [label] = predict(model.QDA,X);
end
if strcmp(lbl{indx},'SVM')
    [label] = predict(model.SVM,X);
end
if strcmp(lbl{indx},'PLSDA')
    Sc_val=Centrerval(double(X),model.PLSDA.Sc_cal);
    Y0 = Sc_val*model.PLSDA.Wstar*model.PLSDA.Q; % Calibration prediction
    label = Decentrerval( Y0, model.PLSDA.Y_cal ); % Uncentering of the prediction
    [~,label] = max(label');
end

ClassifMap=double(reshape(label,size(M,1),size(M,2)));

% Display
figure
ha(1)=subplot(211);
imagesc(d,d(1:size(M,1)),double(ClassifMap))
ha(2)=subplot(212);
imagesc(d,d(1:size(M,1)),RGB.*(0.5/mean(mean(RGB,1),2)))
linkaxes(ha,'xy')
end