function [Cartlab,classworkk,ROIclasswork] = LabelledMapCreation(IM)
% Function to create a labelled map for a discrimination.

% INPUT:
%           IM : Hyperspectral datacube or RGB image of the sample
% OUTPUT:
%           Cartlab : Labelled map
%           classworkk : Class of the labelled pixel 
%           ROIclasswork : Spectra of the labelled pixel 
% This is the Matalb toolbox from the papers:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Jacq, K., Rapuc, W., Benoit, A., Develle, A.-L., Coquin, D., 
% Fanget, B., Perrette, Y.,  Sabatier, P., Wilhelm, B., 
% Debret, M., Arnaud, F., 2019. Sedimentary structures 
% discriminations with hyperspectral imaging on 
% sediment cores. Computers & Geosciences

% Please cite our papers if you use our code for your research.

% RGB creation if IM is an hyperspectral image
if size(IM,3)>3
    % ROI
    ROI=IM;
    if size(ROI,3)==144
        RGB=ROI(:,:,[107 111 124])/10000;
    else if size(ROI,3)==98
            RGB=ROI(:,:,[50 26 13])/10000;
        end
    end
else
    ROI=IM;
    RGB=ROI;
end
m=mean(RGB(:));
RGB=RGB*ceil(0.5/m);
nbpixel=size(ROI,1)*size(ROI,2);

if size(ROI,3)==98||size(ROI,3)==144
    ROI=ROI(:,:,15:end-10);
end

% Initialisation
ROIclass=[];class=[];Mclass=[];
Cartlab=zeros(size(ROI,1),size(ROI,2));
disp('Selection des groupes pour classification supervisée')
roisearch='Yes';
list = {'1','2','3','4','5'};

figure;
imagesc(RGB)

while strcmp(roisearch,'Yes')
    hBox = imrect;
    roiPositioni = wait(hBox);
    ROIid=reshape(RGB(roiPositioni(2):roiPositioni(2)+roiPositioni(4),roiPositioni(1):roiPositioni(1)+roiPositioni(3),:),[],size(RGB,3));
    
    Mid=reshape(ROI(roiPositioni(2):roiPositioni(2)+roiPositioni(4),roiPositioni(1):roiPositioni(1)+roiPositioni(3),:),[],size(ROI,3));
    
    [indx,~] = listdlg('ListString',list);
    if isempty(find(class==indx))
        lab=inputdlg({strcat('Class name ',num2str(indx))},'Labelling');
        lbl{indx}=lab{1,1};
    end
    class=[class; ones(size(ROIid,1),1)*indx];
    Cartlab(roiPositioni(2):roiPositioni(2)+roiPositioni(4),roiPositioni(1):roiPositioni(1)+roiPositioni(3))=indx;
    nbclassi(indx)=length(find(class==indx));
    ROIclass=[ROIclass; ROIid];
    
    Mclass=[Mclass;Mid];
    list{indx}=strcat(lbl{indx},' : ',num2str(nbclassi(indx)/nbpixel),'%');
    
    roisearch=questdlg('Do you want to continue ?','ROI sélection','Yes','No','Yes');
end

classi=unique(class);

classwork=class;

ROIclasswork=Mclass;
indxclasswork=classi;

classworkk=zeros(size(classwork,1),length(indxclasswork));
for i=1:size(classwork,1)
    classworkk(i,classwork(i,1))=1;
end

end