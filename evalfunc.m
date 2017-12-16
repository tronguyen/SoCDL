function evalfunc( data, linksave )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
init_socdl;

disp('EVAL: Recall...');
params.utrain = sprintf('data/%s/coldstart/ctr-data/rvone/train-users.dat', params.data);
params.utest = sprintf('data/%s/coldstart/ctr-data/rvone/test-users.dat', params.data);

mU = dlmread(sprintf('%s/final-U.dat',linksave));
mV = dlmread(sprintf('%s/final-V.dat',linksave));
recall = eval_recall(mU, mV, params);
disp(recall); 
dlmwrite(sprintf('%s/eval_ln%f_lu%f.dat',linksave, params.ln, params.lu),recall,'delimiter',' ');

end

