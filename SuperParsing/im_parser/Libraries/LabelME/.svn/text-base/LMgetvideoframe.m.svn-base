function [newannotation,img,jo] = LMgetvideoframe(annotation, f, HOMEVIDEOS)
%
% Returns the annotation for frame number 'f' as if it is was the
% annotation of a single image.



newannotation = annotation;
jo = [];
img = [];

if isfield(annotation, 'object')
    newannotation = rmfield(newannotation, 'object');
    Nobjects = length(annotation.object);
    m = 0;
    for n = 1:Nobjects
        [X,Y,t] = getLMpolygon(annotation.object(n).polygon);
        i = find(t==f);
        
        if ~isempty(i)
            m = m+1;
            obj = annotation.object(n);
            obj = rmfield(obj, 'polygon');
            obj.polygon.x = X(:,i);
            obj.polygon.y = Y(:,i);
            obj.polygon.t = t(i);
            newannotation.object(m) = obj;
            jo(m) = n;
        end
    end
end


if nargout > 1 && nargin == 3
    videofilename = fullfile(HOMEVIDEOS, annotation.folder, annotation.filename);
    videoobj = mmreader(videofilename);
    img = read(videoobj, f+1);
end

