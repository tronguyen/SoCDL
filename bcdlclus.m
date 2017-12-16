function [MU] = bcdlclus( MU, params, isMC)
if nargin < 3
    isMC = 0;
end
% set seed number
rng(11112);
if ~isMC
    params.daelink = sprintf('data/%s/data_network%s_k%d', params.data, params.rand, params.numwalks);
    load(sprintf('%s/network_order.mat', params.daelink)); % ord
    
else
    params.daelink = sprintf('data/%s/data_network_rating_k%d', params.data, params.numwalks);
    load(sprintf('%s/shuff_order.mat', params.daelink)); % ord 
    u_ord = shuff_ord;
end
ctr_loss = exp(50);
%% Pretrain DAE
for ep = 1:100,
    MU = train_dae_from_ctr(MU, [], params, 2);
end


%% Main-part
% fopen( 'results.txt', 'wt' );
for ep=1:params.maxepoch,
    ctr_loss_old = ctr_loss;
    
    %% Controlling learn-rate
    if ep > 0.15 * params.maxepoch
        MU.lrate = MU.lrate / (1 + ep/params.maxepoch);
%         MU.lrate = MU.lrate / sqrt(ep-100);
    end;
    
    %% Train feature-dae
%     disp('Train DAE - User side ...');
    
%     if ep==1,
%         MU = train_dae_from_ctr(MU, [], params, 2);
%     else
%         m_U = dlmread(sprintf('%s/final-U.dat',params.save));
%         m_U = m_U(u_ord,:);
%         MU = train_dae_from_ctr(MU, m_U, params, 2);
%     end;
    
    % Get dae-hidden
    m_delta = sdae_get_hidden_full(0,MU, params);
    [~, I] = sort(u_ord);
    m_delta = m_delta(I, :);
    
    % Sample the hidden layer
    m_delta = m_delta > rand(size(m_delta));
%     mnets = containers.Map('KeyType','int32', 'ValueType','any');
    for uidx = 1:params.nU,
        uline = sum(m_delta(:,m_delta(uidx,:)==1), 2) > 0;
        uline = find(uline==1);
        uline = [length(uline) uline'];
        dlmwrite(sprintf('%s/net-users.dat',params.save),uline,'delimiter',' ','-append');
    end
   
        
    %% Train CTR: call ctr with init from 2-dae above
    if ep==params.maxepoch
        max_iter = params.maxepochctr;
    else
        max_iter = params.minepochctr;
    end
    
    if ~params.coldstart
        ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-ctrsmf/ctr --max_iter %d --mult data/%s/ctr-data/mult.dat --theta_init %s/final-theta.dat --beta_init %s/final-beta.dat --directory %s --user data/%s/ctr-data/train-users.dat --item data/%s/ctr-data/train-items.dat --net %s/net-users.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --lambda_q %f --lambda_s %f --save_lag 100 --random_seed 123 >> %s/%s', ...
            mlib,max_iter,params.data,params.save,params.save,...
            params.save,params.data,params.data,params.save,...
            params.maxepoch,params.nF, params.lv,params.lu,params.lq,params.ls,...
            params.save,params.ctr_log);
    else
        ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-ctrsmf/ctr --max_iter %d --mult data/%s/ctr-data/mult.dat --theta_init %s/final-theta.dat --beta_init %s/final-beta.dat --directory %s --user data/%s/coldstart/ctr-data/%s/train-users.dat --item data/%s/coldstart/ctr-data/%s/train-items.dat --net %s/net-users.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --lambda_q %f --lambda_s %f --save_lag 100 --random_seed 123 >> %s/%s', ...
            mlib,max_iter,params.data,params.save,params.save,...
            params.save,params.data,params.coldid,params.data,params.coldid,params.save,...
            params.maxepoch,params.nF, params.lv,params.lu,params.lq,params.ls,...
            params.save,params.ctr_log);  
    end

    if mod(ep,params.ctr_ratio)==0 || ep==params.maxepoch || ep<5
        system(ctrcmd);
    end
    
    
    % compute negative log likelihood
    ctr_loss = dlmread(sprintf('%s/final-likelihood.dat',params.save));
    dae_loss = MU.recon_errors(end)*params.ln/2;
    neg_likelihood = ...
        -ctr_loss(1,1)+...
        dae_loss;

    fprintf(1, '- %d/%d - tre/cl/t: %f/%f/%f/%f\n', ep, ...
        params.maxepoch, neg_likelihood,-ctr_loss(1,1), dae_loss,toc);
%     fprintf(myfid, '%d: %d/%d - tre/cl/t: %4.0f/%0.4f/%f\n', pid, ...
%         step, params.maxepoch, neg_likelihood,ctr_loss(1,1),toc);
    
    
    % save tmp result according to save_lag
%     params.save_lag = params.maxepoch;
%     if mod(ep,params.save_lag)==0
%         system(sprintf('cp %s/final-V.dat %s/tmp/%d-V.dat',params.save,...
%             params.save,ep));
%         system(sprintf('cp %s/final-U.dat %s/tmp/%d-U.dat',params.save,...
%             params.save,ep));
%     end
    converge = abs((ctr_loss-ctr_loss_old)/ctr_loss_old);
    if converge < params.earlystop
        break;
    end
end;

end

