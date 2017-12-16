function s=TimeStamp(params)
if ~params.coldstart
    s=strcat(datestr(clock,'yyyymmddHHMMSS'),datestr(clock,'FFF'));
else
    s=strcat('cold', datestr(clock,'yyyymmddHHMMSS'),datestr(clock,'FFF'));
end