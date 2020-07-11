function [dsample,dEevent,TypEvent,nbpeak]=EvenmentChronicalCompXRF(RGB,d,E,C,dsample,dXRF,El1,El2,El3)
% Function to associate HSI chronicle with XRF proxies.
% INPUT:
%      RGB: RGB image of the sample (m*n*3)
%      d: Event depth vector (1*k)
%      E: Event thickness vector (1*k)
%      C: Classification map (m*n)
%      dsample: Associated depth vector (1*n)
%      dXRF: XRF depth vector (1*l)
%      El1: XRF proxy 1 (1*l)
%      El2: XRF proxy 2 (1*l)
%      El3: XRF proxy 3 (1*l)
% OUTPUT:
%      dsample: Depth vector (1*n)
%      dEevent: Event depth vector
%      TypEvent: Event type based on XRF peaks
%      nbpeak: Number of peak inside an event

[~,ia,~] = unique(dXRF);
dXRF=dXRF(ia);
El1=El1(ia);
El2=El2(ia);
El3=El3(ia);

% Vecteur profondeur
if length(dsample)==2
dinit=dsample(1);
dend=dsample(2);
resol=(dend-dinit)/size(RGB,2);
dsample=dinit:resol:dend;
dsample=dsample(2:end);
end

resolXRF=mean(dXRF(2:end)-dXRF(1:end-1));

% vecteur profondeur crueE
dE=zeros(1,size(dsample,2));
for i=1:length(E)
%     [a,b]=find(dsample<(d(i)+0.1*E(i)+0.2)&dsample>(d(i)-E(i)));
    [a,b]=find(dsample<(d(i)+0.1)&dsample>(d(i)-E(i)-0.1));
    dE(a,b)=1;
end

% Lissage
El1_liss=savgol(El1',7,2,0);
El2_liss=savgol(El2',7,2,0);
El3_liss=savgol(El3',7,2,0);

% Affichage
figure;
ha(1)=subplot(311);
imagesc(dsample,dsample(1:size(RGB,1)),C)
set(gca,'fontsize',14)
ha(2)=subplot(312);
plot(dsample,dE,'linewidth',2)
grid on
set(gca,'fontsize',14)
ha(3)=subplot(313);
yyaxis left
plot(dXRF,El1_liss,'linewidth',2);
yyaxis right
plot(dXRF,El2_liss,'linewidth',2);
grid on
grid minor
set(gca,'fontsize',14)
legend('Mn','Ti')
linkaxes(ha,'x')

% figure;
% ha(1)=subplot(211);
% imagesc(dsample,dsample(1:size(RGB,1)),C)
% ha(2)=subplot(212);
% findpeaks(El1_liss,dXRF)%,'MinPeakWidth',.1,'Threshold',15)
% hold on
% plot(dXRF,El2)
% linkaxes(ha,'x')

[pks_El1,locs_El1] = findpeaks(El1_liss);
[pks_El2,locs_El2] = findpeaks(El2_liss);%,'MinPeakWidth',.05,'Threshold',10);
[pks_El3,locs_El3] = findpeaks(El3_liss);

% Flood = Mn + Ti
TypEvent=zeros(1,length(E));
En2=[];
dn2=[];
iterr=0;
for i=1:length(E)
    %     aca=[];bca=[];
    %     [aca,bca]=find((dXRF(locs_Ca))'<(d(i)+0.1*E(i))&(dXRF(locs_Ca))'>(d(i)-1.1*E(i)));
    %     if isempty(aca)==0
    %         TypEvent(i)=5;
    %     else
    a=[];b=[];
%     [a,b]=find(dXRF(locs_El1)'<(d(i))&dXRF(locs_El1)'>(d(i)-E(i)-0.2)); % pic Mn
%     [b,c]=find(dXRF(locs_El2)'<(d(i)+0.1)&dXRF(locs_El2)'>(d(i)-E(i))); % pic Ti
    
    [a,b]=find(dXRF(locs_El1)'<(d(i))&dXRF(locs_El1)'>(d(i)-E(i))); % pic Mn
    [b,c]=find(dXRF(locs_El2)'<(d(i))&dXRF(locs_El2)'>(d(i)-E(i)-.2)); % pic Ti
    
    if isempty(a)==0&&isempty(b)==0
        TypEvent(i)=1;
        nbpeak(i)=length(a);
        %         if nbpeak(i)==8
        %             aaaa=1;
        %         end
        %         if nbpeak(i)>1 % verif nombre de pic de Mn pour le dépot
        %             posMn=dXRF(locs_Mn(dXRF(locs_Mn)'<(d(i)+0.1*E(i))&dXRF(locs_Mn)'>(d(i)-1.1*E(i))));
        %             for j=length(posMn):-1:1
        %                 a=[];b=[];
        %                 [a,b]=find(abs(dsample-posMn(j))==min(abs(dsample-posMn(j))));
        %                 pxlMn=sum(C(round(size(C,1)/10):round(9*size(C,1)/10),b-5:b+5)==2,1);
        %                 iter=0;
        %                 if pxlMn>10
        %                     iter=iter+1;
        %                     ai=[];bi=[];
        %                     [ai,bi]=find(pxlMn>10&max(pxlMn));
        %                     if iter==1 % efface les epaisseurs et profondeur d'avant
        %                         dn2=[dn2 d(i) dsample(b-5+ai(end)-1)];
        %                         En2=[En2 dn2(iter)-dn2(iter+1)];
        %                         iterr=iterr+1;
        %
        %                     else if iter==length(posMn)
        %                             dn2=[dn2 dsample(b-5+ai(end)-1)];
        %                             En2=[En2 dn2(iter)-dn2(iter+1) dn2(iter+1)-(d(i)-E(i))];
        %                         else
        %                             dn2=[dn2 dsample(b-5+ai(end)-1)];
        %                             En2=[En2 dn2(iter)-dn2(iter+1)];
        %                         end
        %                     end
        %                 else if length(posMn)==j
        %                         if iter==0
        %                             dn2=[dn2 d(i)];
        %                             En2=[En2 E(i)];
        %                         else
        %                             En2=[En2 dn2(iter+1)-(d(i)-E(i))];
        %                         end
        %                     end
        %                 end
        %             end
        %         else
        %             dn2=[dn2 d(i)];
        %             En2=[En2 E(i)];
        %         end
    else
        amn=[];bti=[];
        [amn,bmn]=find((dXRF(locs_El1))'<(d(i)+0.1*E(i)+0.2)&(dXRF(locs_El1))'>(d(i)-E(i)));
        if isempty(amn)==0
            TypEvent(i)=2;
            nbpeak(i)=length(amn);
        else
            ati=[];bti=[];
            [ati,bti]=find((dXRF(locs_El2))'<(d(i)+0.1*E(i)+0.2)&(dXRF(locs_El2))'>(d(i)-E(i)));
            if isempty(ati)==0
                TypEvent(i)=3;
                nbpeak(i)=0;
            else 
                TypEvent(i)=0;
                nbpeak(i)=0;
            end
        end
    end
    %     end
end
%
% [chronical_d,chronical_dcum,chronical_ecum]=EvenmentChronical(d(TypEvent==1),E(TypEvent==1),dsample,0.1,10);
% [chronical_d,chronical_dcum,chronical_ecum]=EvenmentChronical(d(TypEvent==2),E(TypEvent==2),dsample,0.1,10);
% [chronical_d,chronical_dcum,chronical_ecum]=EvenmentChronical(d(TypEvent==3),E(TypEvent==3),dsample,0.1,10);

dEevent=zeros(1,size(dsample,2));
for i=1:length(E)
    [a,b]=find(dsample<d(i)&dsample>(d(i)-E(i)));
    dEevent(a,b)=TypEvent(i)+1;
end

figure;
ha(1)=subplot(411);
imagesc(dsample,dsample(1:size(RGB,1)),RGB)
ha(2)=subplot(412);
imagesc(dsample,dsample(1:size(RGB,1)),C)
ha(3)=subplot(413);
plot(dsample,dEevent)
grid on
title('0=rien, 1=?, 2=Crue (El1+2), 3=Séismes(El1), 4=? (El2)')
ha(4)=subplot(414);
[AX,H1,H2]=plotyy(dXRF,El1_liss,dXRF,El2_liss,'plot','Parent',ha(4),'Parent',ha(4));
grid on
grid minor
xlim([dXRF(1) dXRF(end)])
legend('El1','El2')
linkaxes([ha(1:3) AX],'x')

end