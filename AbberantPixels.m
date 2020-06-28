function mask=AbberantPixels(M,RGB,d,wl,fig,lim)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Reed-Xiaoli Detector (RXD)
% RXD=zeros(size(M,1),size(M,2));
% h=waitbar(0,'Perform RX detector');
% mMean = mean(reshape(M,[],size(M,3))', 2);
% sigma = hyperCov(reshape(M,[],size(M,3))');
% for i=1:size(M,2)
%     waitbar(i/size(M,2))
%     [result, ~, ~] = hyperRxDetector(squeeze(M(:,i,:))',mMean,sigma);
%     RXD(:,i)=result;
% end
% close(h)
%
% figure;
% ha(1)=subplot(211);
% imagesc(d,d(1:size(M,1)),RGB)
% colorbar
% ha(2)=subplot(212);
% imagesc(d,d(1:size(M,1)),RXD)
% caxis([median(RXD(:))-3*std(RXD(:)) median(RXD(:))+3*std(RXD(:))])
% colormap(jet)
% colorbar
% linkaxes(ha,'xy')

%% STD
% if sum(wl<500)>1
%     [~,b]=find(abs(wl-950)==min(abs(wl-950)));
%     IM=squeeze(M(:,:,b));
%     if nargin>5
%         if lim>100
%             lim1=lim;
%             lim2=10000;
%         else if lim<100
%                 lim1=median(IM(:))-lim*std(IM(:));
%                 lim2=median(IM(:))+lim*std(IM(:));
%             end
%         end
%     else
%         lim1=median(IM(:))-2*std(IM(:));
%         lim2=median(IM(:))+2*std(IM(:));
%     end
%     IMd=reshape(IM,[],1);
%     maskd=zeros(size(M,1)*size(M,2),1);
%     maskd(IMd<lim1|IMd>lim2)=1;
%     mask=reshape(maskd,size(M,1),size(M,2));
%
%     if nargin>4
%         figure;
%         ha(1)=subplot(311);
%         imagesc(d,d(1:size(M,1)),RGB)
%         set(gca,'fontsize',14)
%         colorbar
%
%         ha(2)=subplot(312);
%         IM=squeeze(M(:,:,b));
%         imagesc(d,d(1:size(M,1)),IM)
%         caxis([median(IM(:))-3*std(IM(:)) median(IM(:))+3*std(IM(:))])
%         colormap(jet)
%         set(gca,'fontsize',14)
%         colorbar
%
%         ha(3)=subplot(313);
%         imagesc(d,d(1:size(M,1)),mask)
%         colormap(jet)
%         colorbar
%         caxis([0 1])
%         set(gca,'fontsize',14)
%         xlabel('Depth (cm)')
%         linkaxes(ha,'xy')
%     end
% else
IM=reshape(median(reshape(M,[],size(M,3)),2),size(M,1),size(M,2));
if nargin>5
    if lim>100
        lim1=lim;
        lim2=10000;
    else if lim<100
            lim1=median(IM(:))-lim*std(IM(:));
            lim2=median(IM(:))+lim*std(IM(:));
        end
    end
else
    lim1=median(IM(:))-2*std(IM(:));
    lim2=median(IM(:))+2*std(IM(:));
end
IMd=reshape(IM,[],1);
maskd=ones(size(M,1)*size(M,2),1);
maskd(IMd<lim1|IMd>lim2)=NaN;
mask=reshape(maskd,size(M,1),size(M,2));

if nargin>4
    if fig>0
        figure;
        ha(1)=subplot(311);
        imagesc(d,d(1:size(M,1)),RGB)
        set(gca,'fontsize',14)
        colorbar
        
        ha(2)=subplot(312);
        imagesc(d,d(1:size(M,1)),IM)
        caxis([median(IM(:))-3*std(IM(:)) median(IM(:))+3*std(IM(:))])
        colormap(jet)
        set(gca,'fontsize',14)
        colorbar
        
        ha(3)=subplot(313);
        imagesc(d,d(1:size(M,1)),mask)
        colormap(jet)
        colorbar
        caxis([0 1])
        set(gca,'fontsize',14)
        xlabel('Depth (cm)')
        linkaxes(ha,'xy')
    end
end
% end

end