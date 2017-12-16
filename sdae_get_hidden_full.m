% sdae_get_hidden
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [mx] = sdae_get_hidden_full(mask_output, S, params, isSM)

if nargin < 4
    isSM = 0;
end

n_samples = S.nsamples;
minibatch_sz = S.batch_size;
numbatches = ceil(n_samples/minibatch_sz);

layers = S.layers;
n_layers = length(layers);

mx = zeros(n_samples, S.layers(end));

for batch=1:numbatches
    load(sprintf('%s/buff_%d.mat', params.daelink, batch));
    h_mf = batchdata;
    for l = 2:n_layers
        h_mf = bsxfun(@plus, h_mf * S.W{l-1}, S.biases{l}');
        if S.dropout~=0 && l~=n_layers
            % recover from dropout
            h_mf = h_mf.*(1-S.dropout);
        end
        if S.dropout~=0 && l==n_layers && mask_output
            % recover from dropout
            h_mf = h_mf.*(1-S.dropout);
        end
        
        if l < n_layers || S.bottleneck.binary
            h_mf = sigmoid(h_mf,S.hidden.use_tanh);
        end
        if isSM
            h_mf = softmax(h_mf);
        end
    end;
    
    mx((batch-1)*minibatch_sz + 1 : min(batch*minibatch_sz, n_samples), :) = h_mf;
end

% if S.bottleneck.binary
%     if target_sparsity > 0
%         avg_acts = mean(h_mf, 1);
%         diff_acts = max(avg_acts - (1 - target_sparsity), 0);
%         h_mf = min(max(bsxfun(@minus, h_mf, diff_acts), 0), 1);
%     end
% end


