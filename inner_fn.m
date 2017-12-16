function [r] = inner_fn(s, mul)
    r_avai = sum(s~=0, 2);
    x = size(s,2) - r_avai;
    y = r_avai * mul;
    y = min([x,y], [], 2);
    ix = randsample(x,y);
    [~,f] = find(s==0);
    s(f(ix)) = 1;
    r = s;
end