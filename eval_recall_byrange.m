function [ res ] = eval_recall_byrange( params )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

params.utrain = sprintf('data/%s/coldstart/ctr-data/%s/train-users.dat', params.data, params.coldid);
params.utest = sprintf('data/%s/coldstart/ctr-data/%s/test-users.dat', params.data, params.coldid);

mU = dlmread(sprintf('%s/final-U.dat',params.save));
mV = dlmread(sprintf('%s/final-V.dat',params.save));

if params.data == 'delicious',
    rng = [5, 10, 20, 25, 30];
elseif params.data == 'lastfm'
    rng = [5, 10, 20, 25];
end
idx_rng = zeros(params.nU,1);

uTrain = mapreader(params.utrain, 1);
uTest = mapreader(params.utest, 1);

uRate = mU*mV';
uRecall = zeros(params.nU, params.topM/20); % store recall
uIgnore = [];
for uid=1:params.nU,
    uRate(uid, uTrain(uid)) = -inf;
    [~, I] = sort(uRate(uid, :), 'descend');
    ugold = uTest(uid);
    if ~isempty(ugold),
        for top=1:params.topM/20,
            C = intersect(ugold, I(1:top*20));
            uRecall(uid, top) = length(C)/length(ugold);
        end;
    else
        uIgnore = [uIgnore uid];
    end;
    %% range distribution
    len = length(uTrain(uid));
    if len <= rng(1)
        idx_rng(uid)=1;
    elseif len > rng(1) && len <= rng(2)
        idx_rng(uid)=2;
    elseif len > rng(2) && len <= rng(3)
        idx_rng(uid)=3;
    elseif len > rng(3) && len <= rng(4)
        idx_rng(uid)=4;
    elseif len > rng(4)
        idx_rng(uid)=5;
    end
        
end;
uRecall(uIgnore, :) = [];
idx_rng(uIgnore, :) = [];
num_rng = zeros(length(rng)+1, 1);
res = zeros(length(rng) + 1, params.topM/20);
for i = 1: length(rng)+1
    res(i,:) = mean(uRecall(idx_rng==i,:),1);
    num_rng(i,:) = sum(idx_rng==i);
end
disp(res);
disp(num_rng);
end

