function [E,d] = EvenmentChronicalDetection(C,resol,threshch)
% Estimate the chronical from the classification map
% INPUT:    C: Classification map
%           resol: Pixel size
%           threshch: Ratio of the width to characterize an event (value
%           between 0 and 1)
% OUTPUT:   E: Thickness chronicle
%           d: Depth chronicle

%% Layer detections
% Cwk=abs(C(round(0.4*size(C,1):0.6*size(C,1)),:)-2);
Cwk=abs(C-2);
Smf=abs(round(sum(medfilt2(Cwk,[11 3]))));
S=abs(round(sum(Cwk)));
tresh=threshch*size(Cwk,1);
yi=zeros(1,length(S));
for i=1:length(S)-1
    if (Smf(i)>tresh&&Smf(i+1)<tresh)
        yi(i)=-1;
    else if(Smf(i)<tresh&&Smf(i+1)>tresh)
        yi(i)=1;
        end
    end
end
[~,locsl]=find(yi==1);
[~,locsr]=find(yi==-1);

% Removes solitary boundaries
locsr=locsr(locsr>locsl(1));
locsl=locsl(locsl<locsr(end));

% Thicknesses estimation
y=zeros(1,size(C,2));
for i=1:length(locsr)
    y(locsl(i):locsr(i))=locsr(i)-locsl(i);
end

% Depth and thickness
d=locsr*resol;
E=(locsr-locsl)*resol;

% Divide large event with a second threshold (0.15)
locslp=[];
locsrp=[];
tresh2=0.15*size(C,1);
for i=1:length(locsl)
   if E(i)>0.5
       Smf2=savgol(sum(C(:,locsr(i)-(locsr(i)-locsl(i)):locsr(i))-1),7,2,0);
       yii=zeros(1,length(Smf2));
       for j=1:length(Smf2)-1
           if (Smf2(j)>tresh2&&Smf2(j+1)<tresh2)
               yii(j)=-1;
           else if (Smf2(j)<tresh2&&Smf2(j+1)>tresh2)
                   yii(j)=1;
               end
           end
       end
       [~,locslj]=find(yii==1);
       [~,locsrj]=find(yii==-1);
       if length(locslj)==1
           locslp=[locslp locsl(i)];
           locsrp=[locsrp locsr(i)];
       else
           locslp=[locslp locsl(i) locsrj(2:end)+locsl(i)];
           locsrp=[locsrp locslj(1:end-1)+locsl(i) locsr(i)];
       end

%        figure
%        subplot(211)
%        imagesc(C(:,locsr(i)-(locsr(i)-locsl(i)):locsr(i))-1)
%        subplot(212)
%        plot(Smf2)
%        grid on
   else
       locslp=[locslp locsl(i)];
       locsrp=[locsrp locsr(i)];
   end
end

% Thickness estimation
y=zeros(1,size(C,2));
for i=1:length(locsrp)
    y(locslp(i):locsrp(i))=locsrp(i)-locslp(i);
end

d=[];E=[];
d=locsrp*resol;
E=(locsrp-locslp)*resol;

figure;
ha(1)=subplot(211);
imagesc(C)
ha(2)=subplot(212);
plot(1:length(y),y*resol)
grid on
linkaxes(ha,'x')

end