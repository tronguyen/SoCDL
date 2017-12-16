function data_walk(numwalk, isrand)
rand('seed',11112);
% A: transition matrix
init_lastfm;
% init_delicious;
% init_livejournal;
if isrand
    fname = sprintf('data_network_random/raw/rd_%d/buff_0', isrand-1);
    fold = sprintf('data_network_random/rd%d_k', isrand);
else
    fname = 'data_network_full';
    fold = 'data_network_k';
end
%% create transition matrix
params.datalink = sprintf('data/%s/%s', params.data, fname);

A = datareader(params, params.numnet);
A = bsxfun(@rdivide, A, sum(A,2));

%% init

D = eye(params.numnet);
walks{1} = D*A;
K = numwalk;
if numwalk~=1,
    X = walks{1} * descfunc(1, K);
else
    X = walks{1};
end
minibatch_sz = params.buff; 
nrow = params.numnet;
numbuff = ceil(nrow/minibatch_sz);

nor = 1;
%% Start walking 
for step=2:K,
    walks{step} = walks{step-1}*A;
    X = X + descfunc(step, K) * walks{step};
    nor = nor + descfunc(step, K);
end;
X = X/nor;
% X = bsxfun(@rdivide,X,max(X,[],2));
% Normalize data
% make it zero-mean
% patches_mean = mean(X, 1);
% X = bsxfun(@minus, X, patches_mean);

% make it unit-variance
% patches_std = std(X, [], 1);
% X = bsxfun(@rdivide, X, patches_std);

%% store data
u_ord = randperm(nrow);
Y = X(u_ord,:);
clear X;
mkdir(sprintf('data/%s/%s%d', params.data, fold, numwalk));
for bid=1:numbuff,
    batchdata = Y((bid-1)*minibatch_sz + 1 : min(bid*minibatch_sz, nrow),:);
    save(sprintf('data/%s/%s%d/buff_%d.mat', params.data, fold, numwalk, bid), 'batchdata');
    clear batchdata;
end;
save(sprintf('data/%s/%s%d/network_order.mat', params.data, fold, numwalk), 'u_ord');
end

