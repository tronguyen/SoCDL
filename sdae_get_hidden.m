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
function [h_mf] = sdae_get_hidden(mask_output, x0, S)

layers = S.layers;
n_layers = length(layers);

h_mf = x0;

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
        h_mf = sigmoid(h_mf, S.hidden.use_tanh);
    end
end



