% default_dbm - 
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [S] = default_sdae_mc (M)
    % data type
    S.data.binary = M.bindata;
    S.recon_errors = [];
    %S.data.binary = 0; % for GDBM

    % bottleneck layer
    S.bottleneck.binary = 1;
    %S.bottleneck.binary = 0;

    % nonlinearity: the name of the variable will change in the later revision
    % 0 - sigmoid
    % 1 - tanh
    % 2 - relu
    S.hidden.use_tanh = 0;
    S.visible.use_tanh = M.actvis; % added by hog

    % learning parameters
    S.lrate = M.lrate;
    S.momentum = M.momentum;
    S.weight_decay = M.decay;
    S.minibatch_sz = M.batchsize;
    S.dropout = M.dropout;
    
    S.layers = M.layers;
    S.batch_size = M.batchsize;
    S.nsamples = M.nsamples;

    % denoising
    S.noise = M.noise;

    % structure
    n_layers = length(M.layers);
    layers = M.layers;
    % initializations
    S.W = cell(n_layers-1, 1);
    S.biases = cell(n_layers, 1);
    for l = 1:n_layers
        S.biases{l} = zeros(layers(l), 1);
        S.grad.biases{l} = zeros(size(S.biases{l}));
        if l < n_layers
            %S.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
            S.W{l} = 0.01 * randn(layers(l), layers(l+1));
            S.grad.W{l} = zeros(size(S.W{l}));
        end      
    end
    
    S.adagrad.use = M.adagrad;
    if M.adagrad,
        disp('Using adagrad ...');
        S.adagrad.epsilon = 1e-8;
        S.adagrad.W = cell(n_layers, 1);
        S.adagrad.biases = cell(n_layers, 1);
        for l = 1:n_layers
            S.adagrad.biases{l} = zeros(layers(l), 1);
            if l < n_layers
                S.adagrad.W{l} = zeros(layers(l), layers(l+1));
            end
        end
    end

end

