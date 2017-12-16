function eval_slice( params, ep )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if ~params.coldstart
    params.utrain = sprintf('data/%s/ctr-data/train-users.dat', params.data);
    params.utest = sprintf('data/%s/ctr-data/test-users.dat', params.data);
else
    params.utrain = sprintf('data/%s/coldstart/ctr-data/%s/train-users.dat', params.data, params.coldid);
    params.utest = sprintf('data/%s/coldstart/ctr-data/%s/test-users.dat', params.data, params.coldid);
end;
mU = dlmread(sprintf('%s/final-U.dat',params.save));
mV = dlmread(sprintf('%s/final-V.dat',params.save));
[recall, mrr] = eval_recall(mU, mV, params);

fprintf(1,'Recall at EP-%d ...\n', ep);
disp(recall);
fprintf(1,'Mrr at EP-%d ...\n', ep);
disp(mrr);

dlmwrite(sprintf('%s/recall_track_ln%f_lu%f_lv%f.dat',params.save, params.ln, params.lu, params.lv),[ep recall],'-append','delimiter',' ');
dlmwrite(sprintf('%s/mrr_track_ln%f_lu%f_lv%f.dat',params.save, params.ln, params.lu, params.lv),[ep mrr],'-append','delimiter',' ');

end

