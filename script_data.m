function script_data(topic)
% init_lastfm;
% init_delicious;
% %% create transition matrix
% params.datalink = sprintf('data/delicious/data_network_full');
% 
% X = datareader(params, params.numnet);
% X = bsxfun(@rdivide,X, sum(X,2));

%% convert rating buff to mat file
% for bid=1:693,
%     params.datalink = sprintf('data/delicious/data_item_java/buff_%d', bid);
%     writewhere = sprintf('data/delicious/data_item/buff_%d.mat', bid);
%     datareader(params, params.numtags, 1, writewhere, 1);
% end;


%% plot graph
% fname = 'data_network_full';
% params.datalink = sprintf('data/lastfm/%s', fname);
% A = datareader(params, params.numnet);
% G = graph(A);
cl = {'b','g','r','c','m','y'};
%% Plot deep
load('/Volumes/DATA/Project/Matlab/socdl/save/pretrain/data/lastfm/K10/MX.mat');
W = MX.W{1};
[~,I]=sort(W(:,topic),'descend');

load('/Volumes/DATA/Project/Matlab/socdl/data/lastfm/data_network_k5/network_order.mat');

figure(topic);
for i=1:3
    index = find(u_ord==I(i));
    bid = floor(index/100) + 1;
    bidx = mod(index,100);
    load(sprintf('/Volumes/DATA/Project/Matlab/socdl/data/lastfm/data_network_k5/buff_%d.mat',bid));
    hold on;
    s = cl(randsample(6,1));
    plot(batchdata(bidx,:),char(s));
    clear batchdata;
end

for i=length(I):-1:(length(I))
    index = find(u_ord==I(i));
    bid = floor(index/100) + 1;
    bidx = mod(index,100);
    load(sprintf('/Volumes/DATA/Project/Matlab/socdl/data/lastfm/data_network_k5/buff_%d.mat',bid));
    hold on;
    plot(batchdata(bidx,:),'k');
    clear batchdata;
end
clear
end