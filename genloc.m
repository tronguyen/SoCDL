function [ loc ] = genloc( alg, params )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if alg==-1 && params.isDeep
        loc = sprintf('save/pretrain/data/%s/vmodel',params.data);
    elseif alg==-1 && ~params.isDeep
        loc = sprintf('save/pretrain/data/%s/vmodel/shallow',params.data);
    elseif alg==0 && params.isDeep
        loc = sprintf('save/pretrain/data/%s/K%d',params.data, params.numwalks);
    elseif alg==0 && ~params.isDeep
        loc = sprintf('save/pretrain/data/%s/shallow/K%d',params.data, params.numwalks);
    elseif alg==2
        loc = sprintf('save/vcdl/%s%s',params.sig, TimeStamp(params));
    elseif alg==1
        loc = sprintf('save/socdl/%s%s',params.sig, TimeStamp(params));
    elseif alg==3
        loc = sprintf('save/ucdl/%s%s',params.sig, TimeStamp(params));
    elseif alg==4
        loc = sprintf('save/pmf/%s%s',params.sig, TimeStamp(params));
    elseif alg==5
        loc = sprintf('save/sorec/%s%s',params.sig, TimeStamp(params));
    elseif alg==6
        loc = sprintf('save/bcdl/%s%s',params.sig, TimeStamp(params));
    elseif alg==8
        loc = sprintf('save/ctr/%s%s',params.sig, TimeStamp(params));
    elseif alg==9
        loc = sprintf('save/ctrsmf/%s%s',params.sig, TimeStamp(params));
    elseif alg==11
        loc = sprintf('save/bcdlhalf/%s%s',params.sig, TimeStamp(params));
    elseif alg==12
        loc = sprintf('save/sorbm-wing/%s%s',params.data, TimeStamp(params));
    elseif alg==13
        loc = sprintf('save/sorbm/%s%s',params.data, TimeStamp(params));
    else
        loc = sprintf('save/trash/%s%s',params.data, TimeStamp(params));
    end;
    
    mkdir(strcat(loc, '/ctr-tmp'));
    mkdir(strcat(loc, '/tmp'));

end

