function [MV] = vcdl( MV, params)

load(sprintf('data/%s/data_item/item_order.mat', params.data)); % ord
v_ord = v_ord + 1;
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
        dlmwrite(sprintf('%s/final-gamma.dat',params.save),m_theta,'delimiter',' ');
        
    %% init for ctr
%     disp('Train CTR - Get U&V ...');
    if ep==1
        dlmwrite(sprintf('%s/final-V.dat',params.save),m_theta,'delimiter',' ');
    end
    
    %% Train CTR: call ctr with init from 2-dae above
    if ep==params.maxepoch
        max_iter = params.maxepochctr;
    else
        max_iter = params.minepochctr;
    end
    ctrcmd = sprintf('export LD_LIBRARY_PATH=%s && ctr-part-cdl/ctr --directory %s --user data/%s/coldstart/ctr-data/%s/train-users.dat --item data/%s/coldstart/ctr-data/%s/train-items.dat --max_iter %d --num_factors %d --lambda_v %f --lambda_u %f --save_lag 100 --random_seed 123 --theta_init %s/final-gamma.dat >> %s/%s', ...
        params.gsl_lib,params.save,params.data,params.coldid,params.data,params.coldid,max_iter,params.nF, params.lv,params.lu,...
        params.save,...
        params.save,params.ctr_log);

    if mod(ep,params.ctr_ratio)==0 || ep==params.maxepoch || ep<5
        system(ctrcmd);
    end
    %% EVAL-SLICE
    if ep >= 100 && mod(ep,100)==0 && ep<params.maxepoch
        eval_slice(params, ep);
        save(sprintf('%s/vDAE.mat',params.save), 'MV');
    end
    ctr_loss = dlmread(sprintf('%s/final-likelihood.dat',params.save));
    dae_loss = MV.recon_errors(end)*params.ln/2;
    neg_likelihood = -ctr_loss(1,1) + dae_loss;
    converge = abs((neg_likelihood-neg_likelihood_old)/neg_likelihood_old);
    fprintf(1, '-%d/%d: %f/%f/%f/%.9f\n', ep, ...
        params.maxepoch, neg_likelihood,-ctr_loss(1,1),dae_loss,converge);
    if converge < params.earlystop && ep >= 100
        break;
    end
    
end;
params.endproc = 1;
end

