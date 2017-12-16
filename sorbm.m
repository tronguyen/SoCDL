function [M] = sorbm( M, params, isperm )
rand('seed',11112);
if params.mc_run
    params.daelink = sprintf('data/%s/data_network_rating%s_k%d', params.data, params.rand, params.numwalks);
elseif params.mv_run
    params.daelink = sprintf('data/%s/data_rating%s_k%d', params.data, params.rand, params.numwalks);
end
neg_likelihood = exp(50);
params.utrain = sprintf('data/%s/data_rating/train-rating.dat', params.data);
params.utest = sprintf('data/%s/data_rating/test-rating.dat', params.data);
for ep=1:params.maxepoch,
    neg_likelihood_old = neg_likelihood;
    
    %% Controlling learn-rate
    if ep>0.8*params.maxepoch
        M.momentum = params.max_momentum;
    elseif ep>0.5*params.maxepoch
        M.momentum = params.mid_momentum;
    end
    %% Train feature-dae
    M = train_dae_from_ctr_withmask(M, params, 2);
        
    %% EVAL-SLICE
    if ep >= 100 && mod(ep,100)==0 && ep<params.maxepoch
        save(sprintf('%s/uDAE.mat',params.save), 'M');
        pred_dae(M, params, ep, isperm);
    end
    
    %% CONVERGE-CHECKING
    % compute negative log likelihood
    dae_loss = M.recon_errors(end);
    neg_likelihood = dae_loss;
  
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
    fprintf(2, '-%d/%d: %f/%.9f\n', ep, ...
        params.maxepoch, neg_likelihood, converge);
    
    if converge < params.earlystop && ep >= 100
        break;
    end
end;

end

