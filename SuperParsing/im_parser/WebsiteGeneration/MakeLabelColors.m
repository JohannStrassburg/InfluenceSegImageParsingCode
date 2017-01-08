function [ labelColor ] = MakeLabelColors( numL )
%MAKELABELCOLORS Summary of this function goes here
%   Detailed explanation goes here
    if(numL>49)
        h = (1/numL):(1/numL):1;
        s = repmat([1 .5],[1 ceil(numL/2)]);s = s(1:numL);
        v = repmat([1 1],[1 ceil(numL/2)]);v = v(1:numL);
        rndx = numL:-1:1;%randperm(numL);
        labelColor = [hsv2rgb([h(rndx)' s(rndx)' v(rndx)']); [0 0 0]];
    else
        PreSetColors = [
            128	0	0	;
            0	128	0	;
            128	128	0	;
            0	0	128	;
            128	0	128	;
            0	128	128	;
            128	128	128	;
            64	0	0	;
            192	0	0	;
            64	128	0	;
            192	128	0	;
            64	0	128	;
            192	0	128	;
            64	128	128	;
            192	128	128	;
            0	64	0	;
            128	64	0	;
            0	192	0	;
            128	192	0	;
            0	64	128	;
            128	64	128	;
            0	192	128	;
            128	192	128	;
            64	64	0	;
            192	64	0	;
            64	192	0	;
            192	192	0	;
            64	64	128	;
            192	64	128	;
            64	192	128	;
            192	192	128	;
            0	0	64	;
            128	0	64	;
            0	128	64	;
            128	128	64	;
            0	0	192	;
            128	0	192	;
            0	128	192	;
            128	128	192	;
            64	0	64	;
            192	0	64	;
            64	128	64	;
            192	128	64	;
            64	0	192	;
            192	0	192	;
            64	128	192	;
            192	128	192	;
            0	64	64	;
            128	64	64	];
        labelColor = [PreSetColors(1:numL,:); [0 0 0]]./255;
    end
end

