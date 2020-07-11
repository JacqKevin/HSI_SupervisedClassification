function EvenmentChronicalComp(RGB,C,dsample,d1,E1,d2,E2)
% Function to compare two chronicles.
% INPUT:
%      RGB: RGB image of the sample (m*n*3)
%      C: Classification map (m*n)
%      dsample: Associated depth vector (1*n)
%      d1: Event depth vector of the chronicle 1 (1*k)
%      E1: Event thickness vector of the chronicle 1 (1*k)
%      d2: Event depth vector of the chronicle 2 (1*l)
%      E2: Event thickness vector of the chronicle 2 (1*l)   

% Depth vector
if length(dsample)==2
    dinit=dsample(1);
    dend=dsample(2);
    resol=(dend-dinit)/size(RGB,2);
    dsample=dinit:resol:dend;
    dsample=dsample(2:end);
end

% Event depth vector
dE1=zeros(1,length(dsample));
for i=1:length(E1)
    a=[];b=[];
    [a,b]=find(dsample<d1(i)&dsample>(d1(i)-E1(i)));
    if isempty(a)==0
    dE1(a,b)=(b(end)-b(1))*median(dsample(2:end)-dsample(1:end-1));
    end
end
dE2=zeros(1,length(dsample));
for i=1:length(E2)
    a=[];b=[];
    [a,b]=find(dsample<(d2(i))&dsample>(d2(i)-E2(i)));
    dE2(a,b)=(b(end)-b(1))*median(dsample(2:end)-dsample(1:end-1));
end

figure
subplot(211)
qqplot(d1,d2)
hold on
plot([0 max([max(d1) max(d2)])],[0 max([max(d1) max(d2)])],'k--')
xlim([0 max([max(d1) max(d2)])])
ylim([0 max([max(d1) max(d2)])])
grid on
title('QQplot of the depth distribution')
set(gca,'fontsize',14)
subplot(212)
qqplot(E1,E2)
hold on
plot([0 max([max(E1) max(E2)])],[0 max([max(E1) max(E2)])],'k--')
xlim([0 max([max(E1) max(E2)])])
ylim([0 max([max(E1) max(E2)])])
grid on
title('QQplot of the thickness distribution')
set(gca,'fontsize',14)

figure;
ha(1)=subplot(411);
imagesc(dsample,dsample(1:size(RGB,1)),RGB)
xlim([dsample(1) dsample(end)])
set(gca,'fontsize',14)
ha(2)=subplot(412);
imagesc(dsample,dsample(1:size(RGB,1)),C)
xlim([dsample(1) dsample(end)])
set(gca,'fontsize',14)
ha(3)=subplot(413);
plot(dsample,dE1)
xlim([dsample(1) dsample(end)])
ylim([0 max([max(dE1) max(dE2)])])
grid on
title('Chronicle 1')
set(gca,'fontsize',14)
ha(4)=subplot(414);
plot(dsample,dE2)
xlim([dsample(1) dsample(end)])
ylim([0 max([max(dE1) max(dE2)])])
grid on
title('Chronicle 2')
linkaxes(ha,'x')
set(gca,'fontsize',14)

figure;
ha(3)=subplot(211);
plot(dsample,dE1)
xlim([dsample(1) dsample(end)])
grid on
xlabel('Depth (cm)')
ylabel('Thickness (cm)')
ylim([0 max([max(dE1) max(dE2)])])
title('(A) Chronicle 1')
set(gca,'fontsize',14)
ha(4)=subplot(212);
plot(dsample,dE2)
xlim([dsample(1) dsample(end)])
ylim([0 max([max(dE1) max(dE2)])])
grid on
title('(B) Chronicle 2')
xlabel('Depth (cm)')
ylabel('Thickness (cm)')
linkaxes(ha,'x')
set(gca,'fontsize',14)

figure;
plot(dsample,dE1,'b','linewidth',2)
hold on
plot(dsample,dE2,'g','linewidth',2)
xlim([dsample(1) dsample(end)])
ylim([0 max([max(dE1) max(dE2)])])
grid on
legend('Chronicle 1','Chronicle 2')
xlabel('Depth (cm)')
ylabel('Thickness (cm)')
linkaxes(ha,'x')
set(gca,'fontsize',14)

end