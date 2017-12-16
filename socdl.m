function [MV, MU] = socdl( MV, MU, params )
% set seed number
rng(11112);
% MV: model feature (dae)
% MU: model social (dae)
load(sprintf('data/%s/data_network%s_k%d/network_order.mat', params.data, params.rand, params.numwalks)); % ord
load(sprintf('data/%s/data_item/item_order.mat', params.data)); % ord
v_ord = v_ord + 1;
ctr_loss = exp(50);
for ep=1:params.maxepoch,
    ctr_loss_old = ctr_loss;
    if ep > 0.15 * params.maxepoch
        MU.lrate = MU.lrate / (1 + ep/params.maxepoch);
%         MU.lrate = MU.lrate / sqrt(ep-100);
    end;
    %% Train feature-dae
%     disp('Train DAE - Item side ...');
    params.daelink = sprintf('data/%s/data_item', params.data);
    
    if ep==1,
        MV = train_dae_from_ctr(MV, [], params, 1);
    else
        m_V = dlmread(sprintf('%s/final-V.dat',params.save));
        m_V = m_V(v_ord,:);
        MV = train_dae_from_ctr(MV, m_V, params, 1);
    end;
    
    % Get dae-hidden
    m_theta = sdae_get_hidden_full(0,MV, params);
    [~, I] = sort(v_ord);
    m_theta = m_theta(I,:); % make original order
    dlmwrite(sprintf('%s/final.gamma',params.save),m_theta,'delimiter',' ');
    
    %% Train network-dae
%     disp('Train DAE - Network side ...');
    params.daelink = sprintf('data/%s/data_network%s_k%d/', params.data, params.rand, params.numwalks);
    
    if ep==1,
        MU = train_dae_from_ctr(MU, [], params, 2);
    else
        m_U = dlmread(sprintf('%s/final-U.dat',params.save));
        m_U = m_U(u_ord,:);
        MU = train_dae_from_ctr(MU, m_U, params, 2);
    end;

    % Get dae-hidden
    m_delta = sdae_get_hidden_full(0,MU, params);
    [~, I] = sort(u_ord);
    m_delta = m_delta(I, :);
    if params.urelax
        m_delta = fw_relax(MU, m_delta); 
    end
    dlmwrite(sprintf('%s/final.delta',params.save),m_delta,'delimiter',' ');
        
    %% init for ctr
%     disp('Train CTR - Get U&V ...');
    if ep==1
        dlmwrite(sprintf('%s/final-V.dat',params.save),m_theta,'delimiter',' ');
        dlmwrite(sprintf('%s/final-U.dat',params.save),m_delta,'delimiter',' ');
    end
    
    %% Train CTR: call ctr with init from 2-dae above
    if ep==params.maxepoch
        max_iter = params.maxepochctr;
    else
        max_iter = params.minepochctr;
    end
    if ~params.coldstart
        ctrcmd = sprintf('export LD_LIBRARY_PATH=/usr/local/lib && ctr-part/ctr --directory %s --user data/%s/ctr-data/train-users.dat --item data/%s/ctr-data/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 --theta_init %s/final.gamma --delta_init %s/final.delta >> %s/%s', ...
            params.save,params.data,params.data,max_iter,params.nF, params.lv,params.lu,...
            params.save,params.save,...
            params.save,params.ctr_log);
    else
        ctrcmd = sprintf('export LD_LIBRARY_PATH=/usr/local/lib && ctr-part/ctr --directory %s --user data/%s/coldstart/ctr-data/%s/train-users.dat --item data/%s/coldstart/ctr-data/%s/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 --theta_init %s/final.gamma --delta_init %s/final.delta >> %s/%s', ...
            params.save,params.data,params.coldid,params.data,params.coldid,max_iter,params.nF, params.lv,params.lu,...
            params.save,params.save,...
            params.save,params.ctr_log);
    end
    if mod(ep,params.ctr_ratio)==0 || ep==params.maxepoch || ep<5
        system(ctrcmd);
    end
    % compute negative log likelihood
    ctr_loss = dlmread(sprintf('%s/final-likelihood.dat',params.save));
    dae_loss_u = MU.recon_errors(end)*params.ln/2;
    dae_loss_v = MV.recon_errors(end)*params.ln/2;
    neg_likelihood = ...
        -ctr_loss(1,1)+...
        dae_loss_u + dae_loss_v;

    fprintf(1, '- %d/%d - tre/ctr/u/v/t: %f/%f/%f/%f/%f\n', ep, ...
        params.maxepoch, neg_likelihood,-ctr_loss(1,1), dae_loss_u, dae_loss_v,toc);
    
    converge = abs((ctr_loss-ctr_loss_old)/ctr_loss_old);
    if converge < params.earlystop
        break;
    end
end;

end

