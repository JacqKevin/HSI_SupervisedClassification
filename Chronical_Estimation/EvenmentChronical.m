function [chronical_d,chronical_dcum,chronical_ecum]=EvenmentChronical(d,E,dIM,interv,intr)
% Function to estimation cumulative chronical.
% INPUT:
%           d: Depth
%           E: Thickness
%       optionnal:
%           dIM: IM depth
%           interv: Moving window
%           intr: Integration interval
% OUTPUT:
%           chronical_d: Depth vector
%           chronical_dcum: Cumulative depth vector
%           chronical_ecum: Cumulative thickness vector

if nargin<3
    prompt = {'Initial depth (cm):','Final depth (cm):','Interval (cm):','Integration (cm):'};
    dlgtitle = 'Chronical parameters';
    dims = [1 35];
    definput = {'0','100','0.1','10'};
    answer = inputdlg(prompt,dlgtitle,dims,definput);
    answer=str2double(answer);
else
    answer=[dIM(1) dIM(end) interv intr];
end

if iscell(d)
    for j=1:length(d)
        N=[];edges=[];
        [N,edges,~] = histcounts(d{j},answer(1):answer(3):answer(2));
        
        for i=1:length(edges)-1
            b=[];
            [~,b]=find(d{j}<edges(i+1)&d{j}>=edges(i));
            t(i,j)=sum(E{j}(b));
        end
        
        edges=edges(2:end);
        
        for i=(answer(4)/answer(3))/2+1:length(N)-(answer(4)/answer(3))/2
            c(i,j)=sum(double(N(i-(answer(4)/answer(3))/2:i+(answer(4)/answer(3))/2)));
            ct(i,j)=sum(t(i-(answer(4)/answer(3))/2:i+(answer(4)/answer(3))/2,j));
        end
    end
else
    dpth=answer(1):answer(3):answer(2);
    dd=zeros(1,length(dpth));
    dt=zeros(1,length(dpth));
    for i=1:length(d)
        [a,b]=find(abs(dpth-d(i))==min(abs(dpth-d(i))));
        dd(b(end))=1;
        dt(b(end))=E(i);
    end
    
    for i=(answer(4)/answer(3))/2+1:length(dd)-(answer(4)/answer(3))/2
        c(i)=sum(double(dd(i-(answer(4)/answer(3))/2:i+(answer(4)/answer(3))/2)));
        ct(i)=sum(dt(i-(answer(4)/answer(3))/2:i+(answer(4)/answer(3))/2));
    end
end

figure;
ha(1)=subplot(211);
plot(dpth((answer(4)/answer(3))/2+1:length(dd)-(answer(4)/answer(3))/2),c((answer(4)/answer(3))/2+1:end),'linewidth',2)
grid on
title(strcat('Event frequency (/',num2str(answer(4)),'cm)'))
set(gca,'fontsize',14,'fontweight','bold')
ha(2)=subplot(212);
plot(dpth((answer(4)/answer(3))/2+1:length(dd)-(answer(4)/answer(3))/2),ct((answer(4)/answer(3))/2+1:end),'linewidth',2)
grid on
title(strcat('Cumulative thickness (/',num2str(answer(4)),'cm)'))
set(gca,'fontsize',14,'fontweight','bold')

chronical_d=dpth((answer(4)/answer(3))/2+1:length(dd)-(answer(4)/answer(3))/2);
chronical_dcum=c((answer(4)/answer(3))/2+1:end);
chronical_ecum=ct((answer(4)/answer(3))/2+1:end);

end