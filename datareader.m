function batchdata = datareader(params, veclen, write, writewhere, offset)

if nargin<3,
    write=0;
    writewhere = '';
    offset=0;
end;

location = params.datalink;
fid = fopen(location);
tline = fgets(fid);
count = 1;

while ischar(tline)
    batchtemp = zeros(1,veclen);
    C = strsplit(tline,',');
    B = str2double(C) + offset;
    
    batchtemp(1,B(2:end)) = 1;
    batchdata(count,:) = batchtemp;
    tline = fgets(fid);
    count = count + 1;
end

fclose(fid);

if write,
    save(writewhere, 'batchdata');
end;

return;

