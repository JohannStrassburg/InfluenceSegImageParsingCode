function [ output_args ] = SaveCurrentFigure( outfile, format )
%SAVECURRENTFIGURE Summary of this function goes here
%   Detailed explanation goes here
if(~exist('format','var'))
    format = '-dpng';
end
set(gcf,'PaperPositionMode','auto');
hold off;
if(~isempty(outfile))
    %saveas(gcf,outfile);
    print(outfile,format,'-r96');%saveas(gcf,outfname);
end
end

