function [MU] = ucdl( MU, params)
% set seed number
rand('seed',11112);
% MV: model feature (dae)
% MU: model social (dae)
load(sprintf('data/%s/data_network%s_k%d/network_order.mat', params.data, params.rand, params.numwalks)); % ord
neg_likelihood = exp(50);
for ep=1:params.maxepoch,
    neg_likelihood_old = neg_likelihood;
    
    %% Controlling learn-rate
    if ep>0.8*params.maxepoch
        MU.momentum = params.max_momentum;
    elseif ep>0.5*params.maxepoch
        MU.momentum = params.mid_momentum;
    end
    %% Train feature-dae
%     disp('Train DAE - User side ...');
    params.daelink = sprintf('data/%s/data_network%s_k%d', params.data, params.rand, params.numwalks);
    
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
    dlmwrite(sprintf('%s/final-delta.dat',params.save),m_delta,'delimiter',' ');
        
    %% init for ctr
%     disp('Train CTR - Get U&V ...');
    if ep==1
        dlmwrite(sprintf('%s/final-U.dat',params.save),m_delta,'delimiter',' ');
    end
    
    %% Train CTR: call ctr with init from 2-dae above
    if ep==params.maxepoch
        max_iter = params.maxepochctr;
%     elseif ep<10
%         max_iter = 2;
    else
        max_iter = params.minepochctr;
    end
    if ~params.coldstart
        ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-ucdl/ctr --directory %s --user data/%s/ctr-data/train-users.dat --item data/%s/ctr-data/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 --delta_init %s/final.delta >> %s/%s', ...
            params.gsl_lib,params.save,params.data,params.data,max_iter,params.nF, params.lv,params.lu,...
            params.save,...
            params.save,params.ctr_log);
    else
        ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-ucdl/ctr --directory %s --user data/%s/coldstart/ctr-data/%s/train-users.dat --item data/%s/coldstart/ctr-data/%s/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 --delta_init %s/final-delta.dat >> %s/%s', ...
            params.gsl_lib,params.save,params.data,params.coldid,params.data,params.coldid,max_iter,params.nF, params.lv,params.lu,...
            params.save,...
            params.save,params.ctr_log);
    end;

    if mod(ep,params.ctr_ratio)==0 || ep==params.maxepoch || ep<5
        system(ctrcmd);
    end
    %% EVAL-SLICE
    if ep >= 100 && mod(ep,100)==0 && ep<params.maxepoch
        eval_slice(params, ep);
        save(sprintf('%s/uDAE.mat',params.save), 'MU');
    end
    
    
    %% compute negative log likelihood
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
    converge = abs((neg_likelihood-neg_likelihood_old)/neg_likelihood_old);
    if converge < params.earlystop && ep >= 100
        break;
    end
end;

end

