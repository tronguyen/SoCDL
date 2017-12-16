function eval_tracking( params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if ~params.coldstart
    params.utrain = sprintf('data/%s/ctr-data/train-users.dat', params.data);
    params.utest = sprintf('data/%s/ctr-data/test-users.dat', params.data);
else
    params.utrain = sprintf('data/%s/coldstart/ctr-data/%s/train-users.dat', params.data, params.coldid);
    params.utest = sprintf('data/%s/coldstart/ctr-data/%s/test-users.dat', params.data, params.coldid);
end;
for i=100:100:900
    fprintf(2,'[LOG] Eval at iter-%d ...\n', i);
    mU = dlmread(sprintf('%s/ctr-tmp/%04d-U.dat',params.save,i));
    mV = dlmread(sprintf('%s/ctr-tmp/%04d-V.dat',params.save,i));
    recall = eval_recall(mU, mV, params);
    disp(recall);
    dlmwrite(sprintf('%s/[log]eval_track_ln%f_lu%f_lv%f.dat',params.save, params.ln, params.lu, params.lv),[i recall],'-append','delimiter',' ');
    clear mU mV;
end
end

