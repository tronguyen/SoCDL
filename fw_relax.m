function [ OF ] = fw_relax(M, F )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% OF = sigmoid(bsxfun(@plus, F * M.rW, M.rb'), 2);
% OF = sigmoid(F * M.rW);
OF = F * M.rW;
end

