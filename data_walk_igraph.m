function data_walk_igraph(numwalk, isrand, isperm)
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
    fold = 'data_rating_k';
    
    frname = 'train-rating.dat';
end
%% create transition matrix
    mkdir(sprintf('data/%s/%s%d', params.data, fold, numwalk));
    
    % rating matrix
    params.datalink = sprintf('data/%s/data_rating/%s', params.data, frname);
    X = datareader(params, params.numdim);

%% store data
minibatch_sz = params.buff; 
nrow = params.numnet;
numbuff = ceil(nrow/minibatch_sz);
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

