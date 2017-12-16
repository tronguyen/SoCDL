function [ M ] = mdae(params, M)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% params.daelink = sprintf('data/%s/data_network_k%d/', params.data, params.numwalks);

fprintf(2, 'doing pretrain from scratch\n');
layers = M.layers;
n_layers = length(layers);
Ds = cell(n_layers - 1, 1);
% mystopping = [-2];
%% create X_pre data
n_samples = M.nsamples;
minibatch_sz = M.batch_size;
numbatches = ceil(n_samples/minibatch_sz);
X_pre = [];
for batch=1:numbatches,
    load(sprintf('%s/buff_%d.mat', params.daelink, batch));
    X_pre = [X_pre; batchdata];
end
H = X_pre;
blayers = [0 1 1 1];
use_tanh = 0;
pretrain_iters = params.pretrainiters;
%% Start training each layer
for l = 1:n_layers-1
    % construct DAE and use default configurations
    D = default_dae (layers(l), layers(l+1));
    
    D.data.binary = blayers(l);
    D.hidden.binary = blayers(l+1);
    
    if use_tanh
        if l > 1
            D.visible.use_tanh = 1;
        else
            D.visible.use_tanh = 0; % added by hog
        end
        D.hidden.use_tanh = 1;
    else
        if D.data.binary
            mH = mean(H, 1)';
            D.vbias = min(max(log(mH./(1 - mH)), -4), 4);
        else
            D.vbias = mean(H, 1)';
        end
    end
    
    D.learning.lrate = M.lrate; % default: 1e-1
    D.learning.lrate0 = 5000;
    D.learning.weight_decay = M.weight_decay;
    D.learning.minibatch_sz = minibatch_sz;
    
    D.valid_min_epochs = 10;
    
    %D.noise.drop = 0.2; deleted by hog
    D.noise.drop = M.noise;   % added by hog
    D.sparsity.cost = 0.1;        % added by hog
    D.noise.level = 0;
    
    %D.adagrad.use = 1;
    %D.adagrad.epsilon = 1e-8;
    D.adagrad.use = M.adagrad.use;
    D.adadelta.use = 0;
    D.adadelta.epsilon = 1e-8;
    D.adadelta.momentum = 0.99;
    
    D.iteration.n_epochs = pretrain_iters;
    
    % save the intermediate data after every epoch
    D.hook.per_epoch = {@save_intermediate, {sprintf('dae_mult_%d.mat', l)}};
    
    % print learining process
    D.verbose = 0;
    % display the progress
    D.debug.do_display = 0;
    
    % train RBM
    fprintf(1, 'Training DAE (%d)\n', l);
    
    tic;
    D = dae (M, D, H);
%     mystopping = [mystopping D.mystopping];
    
%     my.fid = fopen(sprintf('%s.log',my.save),'a');
    fprintf(1, 'Training is done after %f seconds\n', toc);
%     fprintf(my.fid, 'Training is done after %f seconds\n', toc);
%     fclose(my.fid);
    
    H = dae_get_hidden(H, D);
%     H_valid = dae_get_hidden(H_valid, D);
    fprintf(1,'\t***Mean: H=%.9f Std: H= %.9f\n', mean(H(:)), mean(std(H,0,2)));
    
    Ds{l} = D;
end % end of for nlayers
%% copy to M-model
for l = 1:n_layers-1
    M.biases{l+1} = Ds{l}.hbias;
    M.W{l} = Ds{l}.W;
end
M.biases{1} = mean(X_pre, 1)';
end

