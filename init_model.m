%% Init feature model
% disp('Init feature model ...');
% MV = default_sdae(para_V);

%% Init social model
if params.mc_run
    disp('Init combined-model');
    MC =  default_sdae(para_C);
elseif params.mv_run
    disp('Init item-model');
    MV =  default_sdae(para_V);
    if params.v_warmstart
        if params.v_isDeep
            disp('Load pretrained V-deep-model...');
            load(sprintf('save/pretrain/data/%s/vmodel/MX.mat',params.data));
        else
            disp('Load pretrained V-shallow-model...');
            load(sprintf('save/pretrain/data/%s/vmodel/shallow/MX.mat',params.data));
        end

        n_layers = length(MV.layers);
        for l = 1:n_layers
            MV.biases{l} = MX.biases{l};
            if l < n_layers
                MV.W{l} = MX.W{l};
            end      
        end
    clear MX;  
    end
end

disp('Init social model');
MU = default_sdae(para_U);
if params.warmstart && alg==6
    if params.isDeep && params.isNormed
        disp('Load pretrained U-deep-model...');
        load(sprintf('save/pretrain/data/%s/K%d/MX.mat',params.data, params.numwalks));
    elseif params.isDeep && ~params.isNormed
        disp('Load pretrained unnormed-deep-model...');
        load(sprintf('save/pretrain/data/%s/nonorm/K%d/MX.mat',params.data, params.numwalks));
    else
        disp('Load pretrained U-shallow-model...');
        load(sprintf('save/pretrain/data/%s/shallow/K%d/MX.mat',params.data, params.numwalks));
    end
    
    if ~params.mc_run
        n_layers = length(MU.layers);
        for l = 1:n_layers
            MU.biases{l} = MX.biases{l};
            if l < n_layers
                MU.W{l} = MX.W{l};
            end      
        end
    else
        n_layers = length(MC.layers);
        for l = 1:n_layers
            MC.biases{l} = MX.biases{l};
            if l < n_layers
                MC.W{l} = MX.W{l};
            end      
        end
    end
    clear MX;
end


if params.urelax,
    disp('Init relaxing user-part');
%     MU.rW = 2 * sqrt(6)/sqrt(para_U.layers(end)+params.nF) * (rand(para_U.layers(end), params.nF) - 0.5);
    
    MU.rW = eye(params.nF); % init community = topic-space
    MU.grad.rW = zeros(size(MU.rW));
    MU.rb = zeros(para_V.layers(end),1);
    MU.grad.rb = zeros(size(MU.rb));
    if MU.adagrad.use
        MU.adagrad.rW = zeros(size(MU.rW));
        MU.adagrad.rb = zeros(size(MU.rb));
    end
    
end;