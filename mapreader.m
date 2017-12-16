function batchdata = mapreader(datalink, offset, sign)

if nargin < 3
    sign = ' ';
end
batchdata = containers.Map('KeyType','int32', 'ValueType','any');
location = datalink;
fid = fopen(location);
tline = fgets(fid);
u = 1;

while ischar(tline),
    %% Read sparse vector
    E_temp = strsplit(tline, sign);
    E_temp = E_temp(2:end);
    X = [];
    Y = [];
    for k=1:length(E_temp),
        val = str2double(char(E_temp(k))) + offset;
        X = [X val];
    end;
    batchdata(u) = X;
    tline = fgets(fid);
    u = u + 1;
end;
fclose(fid);
    
return;

