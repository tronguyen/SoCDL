function [MU] = bcdl( MU, params)
% set seed number
% rng(11112);
rand('seed',11112);
params.daelink = sprintf('data/%s/data_network%s_k%d', params.data, params.rand, params.numwalks);
load(sprintf('%s/network_order.mat', params.daelink)); % ord
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
%     if params.urelax
%         m_delta = fw_relax(MU, m_delta); 
%     end
        
    %% init for ctr
    if ep==1
        dlmwrite(sprintf('%s/final-U.dat',params.save),m_delta,'delimiter',' ');
        dlmwrite(sprintf('%s/final-V.dat',params.save),zeros(params.nV, params.nF),'delimiter',' ');
    end
    
    %% scale encoder-part
    m_delta = m_delta*params.lbscale;
    
    %% write down encoder
    dlmwrite(sprintf('%s/final-delta.dat',params.save),m_delta,'delimiter',' ');
    
    %% Train CTR: call ctr with init from 2-dae above
    if ep==params.maxepoch
        max_iter = params.maxepochctr;
    else
        max_iter = params.minepochctr;
    end
    if ~params.coldstart
        ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-bcdl/ctr --theta_opt --mult data/%s/ctr-data/mult.dat --theta_init %s/final-theta.dat --beta_init %s/final-beta.dat --directory %s --user data/%s/ctr-data/train-users.dat --item data/%s/ctr-data/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 --delta_init %s/final-delta.dat >> %s/%s', ...
            params.gsl_lib,params.data,params.save,params.save,...
            params.save,params.data,params.data,max_iter,params.nF, params.lv,params.lu,...
            params.save,...
            params.save,params.ctr_log);
    else
        ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-bcdl/ctr --theta_opt --mult data/%s/ctr-data/mult.dat --theta_init %s/final-theta.dat --beta_init %s/final-beta.dat --directory %s --user data/%s/coldstart/ctr-data/%s/train-users.dat --item data/%s/coldstart/ctr-data/%s/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 --delta_init %s/final-delta.dat >> %s/%s', ...
            params.gsl_lib,params.data,params.save,params.save,...
            params.save,params.data,params.coldid,params.data,params.coldid,max_iter,params.nF, params.lv,params.lu,...
            params.save,...
            params.save,params.ctr_log);
        
%         ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-bcdl/ctr --theta_opt --mult data/%s/ctr-data/mult.dat --theta_init %s/final-theta.dat --beta_init %s/final-beta.dat --directory %s --user data/%s/coldstart/ctr-data/%s/train-users.dat --item data/%s/coldstart/ctr-data/%s/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 >> %s/%s', ...
%             params.gsl_lib,params.data,params.save,params.save,...
%             params.save,params.data,params.coldid,params.data,params.coldid,max_iter,params.nF, params.lv,params.lu,...
%             params.save,params.ctr_log);
    end;

    if mod(ep,params.ctr_ratio)==0 || ep==params.maxepoch
        system(ctrcmd);
    end
    
    %% EVAL-SLICE
    if ep >= 50 && mod(ep,50)==0 && ep<params.maxepoch
        eval_slice(params, ep);
        save(sprintf('%s/uDAE.mat',params.save), 'MU');
    end
    %% CONVERGE-CHECKING
    % compute negative log likelihood
    ctr_loss = dlmread(sprintf('%s/final-likelihood.dat',params.save));
    llk = dlmread(sprintf('%s/final-likelihood-ruv.dat',params.save));
    dae_loss = MU.recon_errors(end);
    neg_likelihood = ...
        -ctr_loss(1,1)+...
        dae_loss;
  
%     if 0==0,
%         figure(5);
%         subplot(241); hist(MU.W{1});
%         subplot(242); hist(MU.W{2});
%         subplot(243); hist(MU.biases{1});
%         subplot(244); hist(MU.biases{2});
%         
%         subplot(245); hist(MU.grad.W{1});
%         subplot(246); hist(MU.grad.W{2});
%         subplot(247); hist(MU.grad.biases{1});
%         subplot(248); hist(MU.grad.biases{2});
%     end

    converge = abs((neg_likelihood-neg_likelihood_old)/neg_likelihood_old);
%     fprintf(1, '-%d/%d -nll/ctr/dae/conv: %f/%f[%f:%f:%f]/%f/%f\n', ep, ...
%         params.maxepoch, neg_likelihood,-ctr_loss(1,1),llk(1,1),llk(1,2),llk(1,3), dae_loss,converge);
    fprintf(2, '-%d/%d: %f/%f/%f/%.9f\n', ep, ...
        params.maxepoch, neg_likelihood,-ctr_loss(1,1),dae_loss,converge);
    
    if converge < params.earlystop && ep >= 100
        break;
    end
end;
params.endproc = 1;
end

