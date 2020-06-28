function [Res] = Discrimination_Test(ClassifMap,Cartlab)
% Function to compare a labelled and estimated classifcation map.
%
% INPUT:
%	ClassifMap: Estimated classification map (m*n)
%	Cartlab: Labelled map (m*n)
% OUTPUT:
%	Res: Structure with the results
% This is the Matalb toolbox from the papers:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Jacq, K., Rapuc, W., Benoit, A., Develle, A.-L., Coquin, D., 
% Fanget, B., Perrette, Y.,  Sabatier, P., Wilhelm, B., 
% Debret, M., Arnaud, F., 2019. Sedimentary structures 
% discriminations with hyperspectral imaging on 
% sediment cores. Computers & Geosciences

% Please cite our papers if you use our code for your research.

figure;
ha(1)=subplot(211);
imagesc(ClassifMap)
ha(2)=subplot(212);
imagesc(Cartlab);
linkaxes(ha,'xy')

% Unfold the map to a vector
ClassifMapd=ClassifMap(:);
Cartlabd=Cartlab(:);

% Keep the labelled pixel
ClassifMapd_wk=ClassifMapd(Cartlabd>0);
Cartlabd_wk=Cartlabd(Cartlabd>0);

if sum(Cartlabd_wk==1)>sum(Cartlabd_wk==2)
    [a,~]=find(Cartlabd_wk==2);
    [c,~]=find(Cartlabd_wk==1);
    b=rand(length(c),1);
    [~,idx]=sort(b);
    IDX=[idx(1:length(a));a];
else
    [a,~]=find(Cartlabd_wk==2);
    [c,~]=find(Cartlabd_wk==1);
    b=rand(length(a),1);
    [~,idx]=sort(b);
    IDX=[idx(1:length(c));c];
end

% Confusion matrix
C = confusionmat(ClassifMapd_wk(IDX),Cartlabd_wk(IDX));
Res.Accuracy=sum(diag(C))/sum(C(:)); % Accuracy
Res.Recall=C(1,1)/sum(C(1,:));
Res.Selectivity=C(2,2)/sum(C(2,:));
Res.Precision=C(1,1)/sum(C(:,1));
Res.Fmeasure=2*Res.Precision*Res.Recall/(Res.Recall+Res.Precision);
Res
end