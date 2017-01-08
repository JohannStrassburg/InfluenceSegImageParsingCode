function [video, map] = LMplay(varargin)
% Visualizes the polygons in an image.
%
% LMplay(D, ndx, HOMEVIDEOS)
% LMplay(annotationfilename, videofilename)
%
% To play the annotations only:
% LMplay(annotation)

switch length(varargin)
    case 1
        D = varargin{1}; ndx = 1;
        annotation = D.annotation;
        videofilename = '';

        numFrames = str2num(annotation.numFrames);
    case 2
        annotationfilename = varargin{1};
        videofilename = varargin{2};
        
        D = loadXML(annotationfilename); ndx = 1;
        D = LMvalidobjects(D);
        annotation = D.annotation;
    case 3
        D = varargin{1};
        ndx = varargin{2};

        if ndx>length(D)
            error('Index outside database range')
        end

        HOMEVIDEOS = varargin{3};

        annotation = D(ndx).annotation;
        videofilename = fullfile(HOMEVIDEOS, annotation.folder, annotation.filename);
    otherwise
        error('Too many input arguments.')
end

% Define colors
colors = hsv(15);
numFrames = str2num(annotation.numFrames);

% if ~isempty(videofilename)
%     % Open video object
%     videoobj = mmreader(videofilename);
% 
%     % Get the number of frames.
%     numFrames = get(videoobj, 'numberOfFrames');
% end

%numFrames = 100;
% Draw image
figure;
set(gcf, 'position', [83   912   202   167]);
video = [];
for f = 0:numFrames-1
    if ~isempty(videofilename)
        %frame = read(videoobj, f+1);
        frame = LMvideoread(D, ndx, f+1, HOMEVIDEOS);
    else
        frame = zeros([480 640 3], 'uint8');
    end
    hold off; clf
    image(uint8(frame));
    title(sprintf('frame: %d/%d', f, numFrames))
    axis('off'); axis('equal'); 
    hold on

    [nrows ncols cc] = size(frame);
    
    % Draw each object (only non deleted ones)
    if isfield(annotation, 'object')
        Nobjects = length(annotation.object); n=0;
        for i = Nobjects:-1:1
            n = n+1;
            class{n} = annotation.object(i).name; % get object name
            col = colors(mod(sum(double(class{n})),15)+1, :);
            [X,Y,t] = getLMpolygon(annotation.object(i).polygon);

            LineWidth = 4;
            i = find(t==f);
            if ~isempty(i)
                plot([X(:,i); X(1,i)],[Y(:,i); Y(1,i)], 'LineWidth', LineWidth, 'color', [0 0 0]); hold on
                plot([X(:,i); X(1,i)],[Y(:,i); Y(1,i)], 'LineWidth', LineWidth/2, 'color', col);
                hold on
            end
        end
    end
    axis([1 ncols 1 nrows])
    drawnow
    
    if f ==0
        gf = getframe;
        fr = gf.cdata;
        %fr = imresize(fr, .25, 'bilinear');
        video = zeros([size(fr,1) size(fr,2) 1 numFrames], 'uint8');
        [video(:,:,1,f+1), map] = rgb2ind(fr,.14);
    else
        gf = getframe;
        fr = gf.cdata;
        %fr = imresize(fr, .25, 'bilinear');
        fr = uint8(single(fr*.99));
        video(:,:,1,f+1) = rgb2ind(fr, map);
    end
end

