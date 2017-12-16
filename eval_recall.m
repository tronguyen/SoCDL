function [ res, mrr ] = eval_recall( mU, mV, params )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
uTrain = mapreader(params.utrain, 1);
uTest = mapreader(params.utest, 1);

uRate = mU*mV';
uRecall = zeros(params.nU, params.topM/20); % store recall
uMRR = zeros(params.nU, 1);
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
    uMRR(uid) = mean(1./find(ismember(I,ugold)));
end;
uRecall(uIgnore, :) = [];
uMRR(uIgnore, :) = [];
res = mean(uRecall, 1);
mrr = mean(uMRR);
end

