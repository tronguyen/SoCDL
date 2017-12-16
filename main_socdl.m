function main_socdl (rawalg, data, iseval, gid, lu, lv, sp, ln, lq, isrand)
%% init params
disp('Start learning ...');
init_socdl;
init_model; %% generate MU, MV models 

params.sig = sprintf('%s-alg%d-sp%d-', params.data, rawalg, sp);
params.sp = sp;
if nargin < 3
    iseval = 0;
end

if nargin < 4
    gid = 0;
    params.use_gpu = false;
else
    if gid~=-1
        params.use_gpu = gpuDeviceCount;
    else
        params.use_gpu = false;
        disp('Could not use CUDA');
    end
end;

if nargin > 4
    params.lu = lu;
    params.lv = lv;
    params.ln = params.lu;
end

if nargin > 7
    params.ln = ln;
end
if nargin > 8
    params.lq = lq;
end
if nargin > 9
    fprintf(1, 'Random Network %d \n', isrand); % isrand 1-10
    params.rand = sprintf('_random/rd%d', isrand);
else
    params.rand = '';
end

%% start device
if params.use_gpu,
    gpuDevice(gid);
end;
params.alg = alg;

%% start learning
% try
    params.save = genloc(alg, params);
    params.ctr_log = 'ctr-log';
    params.dae_log = 'dae-log';
    cleanupObj = onCleanup(@() cleanMeUp(params));
    
    save(sprintf('%s/settings.mat', params.save), 'params', 'para_U', 'para_V');
    diary(sprintf('%s/train.log', params.save));
    fprintf(1,'Params [lbd-U: %.4f|lbd-V: %.4f|lbd-N: %.4f|lbd-Q: %.4f|lbd-scale: %.4f]\n',params.lu, params.lv, params.ln, params.lq, params.lbscale);

    tic;
    fprintf(1, 'Staring Sample %d - Loc: %s \n', sp, params.save);
    switch alg
        case -1
            fprintf(2,'Train DAE - Item side ...\n');
            params.daelink = sprintf('data/%s/data_item/', params.data);

            fprintf(2, 'datalink: %s\n', params.daelink);
            MX = MV;
            MX = mdae(params, MX);
            save(sprintf('%s/MX.mat',params.save), 'MX');
            
        case 0 % Not-mine pretrain-code      
            fprintf(2,'Train DAE - Network side - Walks(%d) ...\n',params.numwalks);
            if params.isNormed
                params.daelink = sprintf('data/%s/data_network_k%d/', params.data, params.numwalks);
            else
                params.daelink = sprintf('data/%s/nonorm/data_network_k%d/', params.data, params.numwalks);
            end
            fprintf(2, 'datalink: %s\n', params.daelink);
            MX = MU;
            MX = mdae(params, MX);
            save(sprintf('%s/MX.mat',params.save), 'MX');
        
            
        case 1 % original alg
            disp('Social CDL Running ...');
            [MV, MU] = socdl(MV, MU, params);

            disp('DAE-Itemside: ');
            dlmwrite(sprintf('%s/dae_v.log',params.save), MV.recon_errors, 'delimiter',' ');

            disp('DAE-Networkside: ');
            dlmwrite(sprintf('%s/dae_u.log',params.save), MU.recon_errors, 'delimiter',' ');

        case 2 % CDL org
            disp('ORG CDL-V Running ...');
            MV = vcdl(MV, params);

            disp('DAE-Itemside: ');
            dlmwrite(sprintf('%s/dae_v.log',params.save), MV.recon_errors, 'delimiter',' ');

        case 3 % CDL with user-side
            disp('ORG CDL-U Running ...');
            MU = ucdl(MU, params);

            disp('DAE-Userside: ');
            dlmwrite(sprintf('%s/udae.log',params.save), MU.recon_errors, 'delimiter',' ');
            save(sprintf('%s/final-uDAE.mat',params.save), 'MU');

        case 4 % PMF with confidence
            disp('PMF with confidence Running ...');
            mpmf(params);
            type(sprintf('%s/state.log',params.save));

        case 5 % SoRec model
            disp('SoRec Running ...');
            mpmf(params, 4);
            type(sprintf('%s/state.log',params.save));

        case 6 % CDL with chongwang & dae (U-graph)
            fprintf(2,'CTR(chongwang)-DAE(User graph) Running - Walks(%d)...\n', params.numwalks);
            % copy beta-theta init to save folder
            copyfile(sprintf('data/%s/ctr-data/beta-vector.dat', params.data),sprintf('%s/final-beta.dat', params.save));
            copyfile(sprintf('data/%s/ctr-data/theta-vector.dat', params.data),sprintf('%s/final-theta.dat', params.save));
%             copyfile(sprintf('data/%s/ctr-data/theta-vector.dat', params.data),sprintf('%s/final-V.dat', params.save));

            MU = bcdl(MU, params);
            dlmwrite(sprintf('%s/dae_u.log',params.save), MU.recon_errors, 'delimiter',' ');
            save(sprintf('%s/final-uDAE.mat',params.save), 'MU');

        case 7 % CDL with chongwang & dae( of UI-graph)
            disp('CTR(chongwang)-DAE(UI graph) Running ...');
            % copy beta-theta init to save folder
            copyfile(sprintf('data/%s/ctr-data/beta-vector.dat', params.data),sprintf('%s/final-beta.dat', params.save));
            copyfile(sprintf('data/%s/ctr-data/theta-vector.dat', params.data),sprintf('%s/final-theta.dat', params.save));
            copyfile(sprintf('data/%s/ctr-data/theta-vector.dat', params.data),sprintf('%s/final-V.dat', params.save));

            MC = bcdl(MC, params);
            disp('DAE-Userside: ');
            dlmwrite(sprintf('%s/dae_c.log',params.save), MC.recon_errors, 'delimiter',' ');

        case 8 % CTR-chongwang
            disp('CTR(chongwang) Running ...');
            copyfile(sprintf('data/%s/ctr-data/beta-vector.dat', params.data),sprintf('%s/final-beta.dat', params.save));
            copyfile(sprintf('data/%s/ctr-data/theta-vector.dat', params.data),sprintf('%s/final-theta.dat', params.save));
            mpmf(params, 2);
            type(sprintf('%s/state.log',params.save));

        case 9 % CTR-SMF paper: my implementation
            disp('CTR-SMF Running ...');
            copyfile(sprintf('data/%s/ctr-data/beta-vector.dat', params.data),sprintf('%s/final-beta.dat', params.save));
            copyfile(sprintf('data/%s/ctr-data/theta-vector.dat', params.data),sprintf('%s/final-theta.dat', params.save));
            mpmf(params, 3);
            type(sprintf('%s/state.log',params.save));

        case 10 % CTR-SMF-CLUS
            disp('CTR-SMF by DAE-CLUSTERING...');
            MU = bcdlclus(MU, params);

        case 11 % half-half model
            disp('BCDL-Half Running ...');
            % copy beta-theta init to save folder
            copyfile(sprintf('data/%s/ctr-data/beta-vector-100.dat', params.data),sprintf('%s/final-beta.dat', params.save));
            copyfile(sprintf('data/%s/ctr-data/theta-vector-100.dat', params.data),sprintf('%s/final-theta.dat', params.save));

            tmp = dlmread(sprintf('data/%s/ctr-data/theta-vector-100.dat', params.data));
            tmp = [tmp,zeros(params.nV, params.nF/2)];

            dlmwrite(sprintf('%s/final-V.dat', params.save), tmp, 'delimiter',' ');
            MU = bcdl_half(MU, params);

            disp('DAE-Userside: ');
            dlmwrite(sprintf('%s/dae_u.log',params.save), MU.recon_errors, 'delimiter',' ');
            save(sprintf('%s/uDAE.mat',params.save), 'MU');

%         case 13 % pretrain for recsys
%             fprintf(2,'Train DAE - Network side - Walks(%d) ...\n',params.numwalks);
%             params.daelink = sprintf('data/%s/data_network_rating_k%d/', params.data, params.numwalks);
%             MX = MC;
%             MX = mdae(params, MX);
%             save(sprintf('%s/MX.mat',params.save), 'MX');

        case 12 % revise recsys paper
            fprintf(2,'SoRBM-Wing with Neg-Sampling Running - Walk(%d) ...\n', params.numwalks);               
            MC = sorbm(MC, params, 0);
            iseval = 0;
            
        case 13 % revise recsys paper
            fprintf(2,'SoRBM with Neg-Sampling Running - Walk(%d) ...\n', params.numwalks);               
            MV = sorbm(MV, params, 0);
            iseval = 0;
        
                
    end;
    %% EVAL
    if iseval,
        disp('Start evaluating ...');
        if ~params.coldstart
            params.utrain = sprintf('data/%s/ctr-data/sp%d/train-users.dat', params.data, sp);
            params.utest = sprintf('data/%s/ctr-data/sp%d/test-users.dat', params.data, sp);
        else
            params.utrain = sprintf('data/%s/coldstart/ctr-data/%s/train-users.dat', params.data, params.coldid);
            params.utest = sprintf('data/%s/coldstart/ctr-data/%s/test-users.dat', params.data, params.coldid);
        end;
        mU = dlmread(sprintf('%s/final-U.dat',params.save));
        mV = dlmread(sprintf('%s/final-V.dat',params.save));
        [recall, mrr] = eval_recall(mU, mV, params);
        disp(recall);
        disp(mrr);
        dlmwrite(sprintf('%s/[sp%d]recall_ln%f_lu%f_lv%f.dat',params.save, sp, params.ln, params.lu, params.lv),recall,'delimiter',' ');
        dlmwrite(sprintf('%s/[sp%d]mrr_ln%f_lu%f_lv%f.dat',params.save, sp, params.ln, params.lu, params.lv),mrr,'delimiter',' ');
    end;
    toc;
    diary off;
% catch errs
%     disp(['Terminated-Errors: ' errs.identifier]);
% %     exit(1);
% end;
    

end
function cleanMeUp(params)
    if ~params.endproc
        fprintf('cleaning directory...\n');
        rmdir(sprintf('./%s',params.save),'s');
    end;
end
