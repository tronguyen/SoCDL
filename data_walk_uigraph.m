function data_walk_uigraph(numwalk, isrand, isperm)
% set seed number
% rng(11112);
% A: transition matrix
init_lastfm;
% init_delicious;
if isrand
    fnname = 'data_network_full_random';
    fold = 'data_network_random_k';
else
    fnname = 'data_network_full';
    fold = 'data_network_rating_k';
    
    frname = 'train-rating.dat';
end
%% create transition matrix
    mkdir(sprintf('data/%s/%s%d', params.data, fold, numwalk));
    
    % network matrix
    params.datalink = sprintf('data/%s/%s', params.data, fnname);
    C = datareader(params, params.numnet);
    
    % rating matrix
    params.datalink = sprintf('data/%s/data_rating/%s', params.data, frname);
    B = datareader(params, params.numdim);

    % combine B & C matrix
    n = params.numnet;
    m = params.numdim;
    A = eye(n+m);
    A(1:n,:) = [C,B];
    A((n+1):end,1:n) = B';
    
    clear B C;
    
    % normalize matrix A
    A = bsxfun(@rdivide, A, sum(A,2));

%% init

D = [eye(n),zeros(n,m)];
walks{1} = D*A;
K = numwalk;
if numwalk~=1,
    X = walks{1} * descfunc(1, K);
else
    X = walks{1};
end
minibatch_sz = params.buff; 
nrow = n;
numbuff = ceil(nrow/minibatch_sz);
nor = 1;
%% Start walking 
for step=2:K,
    walks{step} = walks{step-1}*A;
    X = X + descfunc(step, K) * walks{step};
    nor = nor + descfunc(step, K);
end;
X = bsxfun(@rdivide,X,max(X,[],2));
% X = X/nor;
% normalize the isolated nodes
% patches_mean = mean(X, 1);
% X = bsxfun(@minus, X, patches_mean);

% make it unit-variance
% patches_std = std(X, [], 1);
% X = bsxfun(@rdivide, X, patches_std);

%% store data
Y = X;
if isperm
    shuff_ord = randperm(nrow);
    Y = X(shuff_ord,:);
end
clear X;

for bid=1:numbuff,
    fprintf(1,'batch-%d\n', bid);
    batchdata = Y((bid-1)*minibatch_sz + 1 : min(bid*minibatch_sz, nrow),:);
    save(sprintf('data/%s/%s%d/buff_%d.mat', params.data, fold, numwalk, bid), 'batchdata');
    clear batchdata;
end;
save(sprintf('data/%s/%s%d/shuff_order.mat', params.data, fold, numwalk), 'shuff_ord');
end

