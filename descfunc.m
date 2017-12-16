function [ val ] = descfunc( t, K )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% val = 1 - 1/K * t/sqrt(K - t + 1); % default

val = 1 + 1/K - t/K; % from pp

% val = 1 + sqrt(1/K) - sqrt(t/K); % for second folder

% val = 1 + log(1/K+1) - log(t/K+1);

% val = 1 + 1/K*(1-t)/sqrt(K - t + 1); 
end

