function [ M ] = train_dae_from_ctr(M, F, params, side)
% F: output U/V from CTR
% M: DAE model
% D: dataset

%% Reading structure
totalerrs = 0;
n_layers = length(M.layers);
n_samples = M.nsamples;
minibatch_sz = M.batch_size;
numbatches = ceil(n_samples/minibatch_sz);
if side==1
    lenc = params.lv;
else
    lenc = params.lu;
end;
%% Start training
for batch=1:numbatches,
%% Forward part
%     fprintf(1, 'load buff-%d\n', batch);
    load(sprintf('%s/buff_%d.mat', params.daelink, batch));
    v0 = batchdata;
    % add noise
    v0_clean = v0;
 
%     if M.data.binary == 0 && M.noise > 0
%         v0 = v0 + M.noise * randn(size(v0));
%     end
    
    if M.noise > 0
        mask = binornd(1, 1 - M.noise, size(v0));
        v0 = v0 .* mask;
        clear mask;
    end
    

    
    if params.use_gpu,
        v0 = gpuArray(single(v0));
        v0_clean = gpuArray(single(v0_clean));
        for l = 1:n_layers
            if l < n_layers 
                M.W{l} = gpuArray(single(M.W{l}));
                M.grad.W{l} = gpuArray(single(M.grad.W{l}));
            end
            M.biases{l} = gpuArray(single(M.biases{l}));
            M.grad.biases{l} = gpuArray(single(M.grad.biases{l}));
        end
        if params.urelax,
            M.rW = gpuArray(single(M.rW));
            M.rb = gpuArray(single(M.rb));
            M.grad.rW = gpuArray(single(M.grad.rW));
            M.grad.rb = gpuArray(single(M.grad.rb));
        end;
        if M.adagrad.use 
            for l = 1:n_layers
                if l < n_layers 
                    M.adagrad.W{l} = gpuArray(single(M.adagrad.W{l}));
                end
                M.adagrad.biases{l} = gpuArray(single(M.adagrad.biases{l}));
            end
        end
    end
    
    h0e = cell(n_layers, 1);
    h0e{1} = v0;
    % FW encode-part
    for l = 2:n_layers
        h0e{l} = bsxfun(@plus, h0e{l-1} * M.W{l-1}, M.biases{l}');

        if l < n_layers || M.bottleneck.binary
            h0e{l} = sigmoid(h0e{l}, M.hidden.use_tanh);
        end
        % add dropout
        if M.dropout~=0
            mask = binornd(1,1-M.dropout,size(h0e{l}));
            h0e{l} = h0e{l}.*mask;
            clear mask;
        end
    end
        
    %%
%     if batch==1,
%         hid = mean(h0e{end},1);
%         fprintf(1,'\t***Mean: H=%.9f Std: H= %.9f\n', mean(hid(:)), mean(std(hid,0,2)));
%     end

    % FW decode-part
    h0d = cell(n_layers, 1);
    h0d{end} = h0e{end};

    for l = n_layers-1:-1:1
        h0d{l} = bsxfun(@plus, h0d{l+1} * M.W{l}', M.biases{l}');
        if l == 1 && M.data.binary
            h0d{l} = sigmoid(h0d{l}, M.visible.use_tanh);
        end
        if l > 1
            h0d{l} = sigmoid(h0d{l}, M.hidden.use_tanh);
        end
        % add dropout, mask every layer except for the last
        if M.dropout~=0 && l~=1
            mask = binornd(1,1-M.dropout,size(h0d{l}));
            h0d{l} = h0d{l}.*mask;
            clear mask;
        end
    end
    if side ==2 && params.urelax
        h0re = sigmoid(bsxfun(@plus, h0e{end} * M.rW, M.rb'), 5);
%         h0re = fw_relax(M, h0e{end});
    end;

%% Compute reconstruction error

    hr = sdae_get_hidden(1, v0_clean, M);
    vr = sdae_get_visible(hr, M);

    if M.data.binary
        rerr = -mean(sum(v0_clean .* log(max(vr, 1e-16)) + (1 - v0_clean) .* log(max(1 - vr, 1e-16)), 2));
    else
        rerr = sum(sum((v0_clean - vr).^2,2));
    end
    if params.use_gpu > 0
        rerr = gather(rerr);
    end
    totalerrs = totalerrs + rerr;


%% Backward part
    % reset gradients
    if side ==2 && params.urelax
        temp.grad.rW = 0 * M.grad.rW;
        temp.grad.rb = 0 * (M.grad.rb)';
    end
    for l = 1:n_layers
        temp.grad.biases{l} = 0 * (M.grad.biases{l})';
        if l < n_layers
            temp.grad.W{l} = 0 * M.grad.W{l};
        end
    end

    % BW from DAE
    deltad = cell(n_layers, 1);
    deltad{1} = (h0d{1} - v0_clean);%.*dsigmoid(h0d{1}, M.visible.use_tanh);
    temp.grad.biases{1} = mean(deltad{1}, 1);
    
    for l = 2:n_layers
        deltad{l} = deltad{l-1} * M.W{l-1};
        if l < n_layers || M.bottleneck.binary
            deltad{l} = deltad{l} .* dsigmoid(h0d{l}, M.hidden.use_tanh);
        end
        temp.grad.biases{l} = mean(deltad{l}, 1);
        temp.grad.W{l-1} = (deltad{l-1}' * h0d{l}) / (size(v0, 1));
    end
    
    deltae = cell(n_layers, 1);
    deltae{end} = deltad{end};
    
    for l = n_layers-1:-1:1
        deltae{l} = deltae{l+1} * M.W{l}';
        if l == 1 && M.data.binary
            if M.hidden.use_tanh==1
                deltae{l} = deltae{l} .* dsigmoid(h0e{l},...
                    M.hidden.use_tanh); % added for tanh by hog
            else
                deltae{l} = deltae{l} .* dsigmoid(h0e{l});
            end
        end
        if l > 1
            deltae{l} = deltae{l} .* dsigmoid(h0e{l}, M.hidden.use_tanh);
            temp.grad.biases{l} = temp.grad.biases{l} + mean(deltae{l}, 1);
        end
        temp.grad.W{l} = temp.grad.W{l} + (h0e{l}' * deltae{l+1}) / (size(v0, 1));
    end
    
%     % FW again  dont fw again
%     h0e = cell(n_layers, 1);
%     h0e{1} = v0;
%     
%     % FW encode-part
%     for l = 2:n_layers
%         h0e{l} = bsxfun(@plus, h0e{l-1} * M.W{l-1}, M.biases{l}');
% 
%         if l < n_layers || M.bottleneck.binary
%             h0e{l} = sigmoid(h0e{l}, M.hidden.use_tanh);
%         end
%         % add dropout
%         if M.dropout~=0
%             mask = binornd(1,1-M.dropout,size(h0e{l}));
%             h0e{l} = h0e{l}.*mask;
%             clear mask;
%         end
%     end
    
    %% CTR PART FROM HERE
%     sm_h0e = softmax(h0e{end}); % do it coz of using softmax for last-hidden
    % FW relax part
%     if side ==2 && params.urelax
%         h0re = sigmoid(bsxfun(@plus, h0h{end} * M.rW, M.rb'), 2);
%         h0rh = fw_relax(M, h0e{end}); % or softmax: 1=tanh
%     end;
    
    
    % BW from CTR (F = U or V)
    if ~isempty(F),
        if params.alg ~=11,
            v_v = F((batch-1)*minibatch_sz + 1 : min(batch*minibatch_sz, n_samples),:);
        else
            v_v = F((batch-1)*minibatch_sz + 1 : min(batch*minibatch_sz, n_samples),params.nF/2+1:end);
        end
        
        
    elseif side ==2 && params.urelax
        v_v = h0re;
    else 
        v_v = h0e{end};
%         v_v = sm_h0e;
    end;
    
    %% 
    if params.use_gpu,
        v_v = gpuArray(single(v_v));
    end;
    
    
    deltah = cell(n_layers,1);
    % BW from relax-side 
    if side ==2 && params.urelax % for user-side relax the mapping
        deltarh = (h0re - v_v).*dsigmoid(h0re,5); % *dsigmoid for sigmoid, softmax noneed
        deltah{end} = deltarh*M.rW';
%         deltah{end} = deltah{end}.*dsigmoid(h0e{end}); % just add to be similar to non-relax
        
        temp.grad.rW = lenc/params.ln*...
            (h0e{end}'*deltarh)/(size(v0,1));
        temp.grad.rb = lenc/params.ln*mean(deltarh,1);
    else
        deltah{end} = (h0e{end}*params.lbscale-v_v)*params.lbscale;
%         deltah{end} = (sm_h0e-v_v);
    end;
        
%     if S.hidden.use_tanh
%         deltah{end} = deltah{end}+1;
%     end
    
    for l = n_layers-1:-1:1
        if l~=n_layers-1
            deltah{l} = deltah{l+1}*M.W{l+1}';
        else
            deltah{l} = deltah{l+1};
        end
        if l==1 && M.data.binary
            if M.hidden.use_tanh==1
                deltah{l} = deltah{l}.*dsigmoid(h0e{l+1},...
                    S.hidden.use_tanh); % added for tanh by hog
            else
                deltah{l} = deltah{l}.*dsigmoid(h0e{l+1});
            end
        end
        if l>1
            deltah{l} = deltah{l}.*dsigmoid(h0e{l+1},M.hidden.use_tanh);
        end
        temp.grad.biases{l+1} = temp.grad.biases{l+1} + lenc/params.ln*mean(deltah{l},1);
        temp.grad.W{l} = temp.grad.W{l} + lenc/params.ln*...
            (h0e{l}'*deltah{l})/(size(v0,1));
    end
    clear v0 h0d h0e h0h v0_clean;
    clear deltah deltae deltad;
    if side ==2 && params.urelax
        clear deltarh h0re;
    end
%% Update model
if M.adagrad.use,
    % update
    for l = 1:n_layers
        M.grad.biases{l} = (1 - M.momentum) * temp.grad.biases{l}' + M.momentum * M.grad.biases{l};
        if l < n_layers
            M.grad.W{l} = (1 - M.momentum) * temp.grad.W{l} + M.momentum * M.grad.W{l};
        end
    end
    
    for l = 1:n_layers
        if l < n_layers
            M.adagrad.W{l} = M.adagrad.W{l} + M.grad.W{l}.^2;
        end
        M.adagrad.biases{l} = M.adagrad.biases{l} + M.grad.biases{l}.^2;
    end
    
    if side ==2 && params.urelax,
        M.adagrad.rW = M.adagrad.rW + M.grad.rW.^2;    
        M.grad.rW = (1 - M.momentum) * temp.grad.rW + M.momentum * M.grad.rW;
        M.rW = M.rW - M.lrate * (M.grad.rW + M.weight_decay * M.rW)./sqrt(M.adagrad.rW + M.adagrad.epsilon);
        
        M.adagrad.rb = M.adagrad.rb + M.grad.rb.^2;    
        M.grad.rb = (1 - M.momentum) * temp.grad.rb' + M.momentum * M.grad.rb;
        M.rb = M.rb - M.lrate * (M.grad.rb + M.weight_decay * M.rb)./sqrt(M.adagrad.rb + M.adagrad.epsilon);
    end
    
    for l = 1:n_layers
        M.biases{l} = M.biases{l} - M.lrate * (M.grad.biases{l} + ...
            M.weight_decay * M.biases{l})./ sqrt(M.adagrad.biases{l} + M.adagrad.epsilon);
        if l < n_layers
            M.W{l} = M.W{l} - M.lrate * (M.grad.W{l} + ...
                M.weight_decay * M.W{l}) ./ sqrt(M.adagrad.W{l} + M.adagrad.epsilon);
        end
    end
    
else
    for l = 1:n_layers
        M.grad.biases{l} = (1 - M.momentum) * temp.grad.biases{l}' + M.momentum * M.grad.biases{l};
        if l < n_layers
            M.grad.W{l} = (1 - M.momentum) * temp.grad.W{l} + M.momentum * M.grad.W{l};
        end
    end
    if side ==2 && params.urelax,
        M.grad.rW = (1 - M.momentum) * temp.grad.rW + M.momentum * M.grad.rW;
        M.rW = M.rW - M.lrate * (M.grad.rW + M.weight_decay * M.rW);
    end

    for l = 1:n_layers
        M.biases{l} = M.biases{l} - M.lrate * (M.grad.biases{l} + M.weight_decay * M.biases{l});
        if l < n_layers
            M.W{l} = M.W{l} - M.lrate * (M.grad.W{l} + M.weight_decay * M.W{l});
        end
    end
end;    
    clear temp;
end;
M.recon_errors = [M.recon_errors totalerrs];
%% Gather data from gpu
if params.use_gpu > 0
    for l = 1:n_layers
        if l < n_layers
            M.W{l} = gather(M.W{l});
            M.grad.W{l} = gather(M.grad.W{l});
        end
        M.biases{l} = gather(M.biases{l});
        M.grad.biases{l} = gather(M.grad.biases{l});
    end
    if side ==2 && params.urelax,
        M.rW = gather(M.rW);
        M.grad.rW = gather(M.grad.rW);
        M.rb = gather(M.rb);
        M.grad.rb = gather(M.grad.rb);
    end
    if M.adagrad.use
        for l = 1:n_layers
            if l < n_layers
                M.adagrad.W{l} = gather(M.adagrad.W{l});
            end
            M.adagrad.biases{l} = gather(M.adagrad.biases{l});
        end
    end
    M.recon_errors = gather(M.recon_errors);
end;

end

