function [ res ] = pred_dae(M, params, ep, isperm)
% F: output U/V from CTR
% M: DAE model
% D: dataset
fprintf(1,'Eval at EP-%d ...\n', ep);
%% Reading structure
n_samples = M.nsamples;
minibatch_sz = M.batch_size;
numbatches = ceil(n_samples/minibatch_sz);

uRate = zeros(n_samples, params.nV);
%% Start training
for batch=1:numbatches,
%% Forward part
%     fprintf(1, 'load buff-%d\n', batch);
    load(sprintf('%s/buff_%d.mat', params.daelink, batch));
    v0_clean = batchdata;
    
    hr = sdae_get_hidden(1, v0_clean, M);
    vr = sdae_get_visible(hr, M, 1);
    if params.mc_run
        uRate((batch-1)*minibatch_sz + 1 : min(batch*minibatch_sz, n_samples), :) = vr(:,params.nU+1:end);
    elseif params.mv_run
        uRate((batch-1)*minibatch_sz + 1 : min(batch*minibatch_sz, n_samples), :) = vr;
    end
end
% revert-shuffle
if isperm,
    load(sprintf('data/%s/data_network_rating_k%d/shuff_order.mat', params.data, params.numwalks)); % ord
    [~, K] = sort(shuff_ord);
    uRate = uRate(K, :);
end

% eval
uTest = mapreader(params.utest, 0, ',');
uTrain = mapreader(params.utrain, 0, ',');

uRecall = zeros(params.nU, params.topM/20); % store recall
uIgnore = [];
for uid=1:params.nU,
    uRate(uid, uTrain(uid)) = -inf;
    [~, I] = sort(uRate(uid, :), 'descend');
    ugold = uTest(uid);
    if ~isempty(ugold),
        for top=1:params.topM/20,
            C = intersect(ugold, I(1:top*20));
            uRecall(uid, top) = length(C)/length(ugold);
        end;
    else
        uIgnore = [uIgnore uid];
    end;
end;
uRecall(uIgnore, :) = [];
res = mean(uRecall, 1);
disp(res);
dlmwrite(sprintf('%s/eval_tracking.dat',params.save),[ep res],'-append','delimiter',' ');

end


