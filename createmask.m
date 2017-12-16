function [ mymask ] = createmask( v0, params )
%create mask for output: negative sampling

% mymask = [binornd(1, params.neg_noise_net, [size(v0,1), params.nU]), ...
%         binornd(1, params.neg_noise_rating, [size(v0,1), params.nV])] + (v0~=0);
% mymask = mymask > 0;

% binarize matrix v0
v0 = v0~=0;  
% inner function
function [r] = m_inner_fn(s, mul)
    r_avai = sum(s~=0, 2);
    x = size(s,2) - r_avai;
    y = r_avai * mul;
    y = min([x,y], [], 2);
    ix = randsample(x,y);
    [~,f] = find(s==0);
    s(f(ix)) = 1;
    r = s;
end

net_part = v0(:,1:params.nU);
temp_net = num2cell(net_part,2);

rat_part = v0(:,params.nU+1:end);
temp_rat = num2cell(rat_part,2);

netmask = cellfun(@(s)m_inner_fn(s, params.neg_noise_net),temp_net,'UniformOutput', false);
ratmask = cellfun(@(s)m_inner_fn(s, params.neg_noise_rat),temp_rat,'UniformOutput', false);

mymask = [cell2mat(netmask), cell2mat(ratmask)];
end

