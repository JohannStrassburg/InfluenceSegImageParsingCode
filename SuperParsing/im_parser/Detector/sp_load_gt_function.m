function extras = sp_load_gt_function(dataset_params, ...
                                        Iname, ...
                                        boxes)
% Given an image, load the corresponding ground truth and compare
% its overlap with the detections inside [boxes].
%
% During detection, this is sometimes the "extra" information
% regarding the overlaps with the boxes inside this image, which is
% scored along the detection scores.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if isnumeric(Iname) == 1
  extras = [];
  return;
else 
  if isstr(Iname)
    %extract image id from the filename (this can be done for VOC
    %images)
    %NOTE(TJM): this is the old way of doing things
    %[tmp,curid,tmp] = fileparts(Iname);
    
    % get GT objects for this image
    ind = strfind(Iname,'Images');
    HOME = Iname(1:ind-1);
    HOMEIMAGES = fullfile(HOME,'Images');
    HOMELABELSET = fullfile(HOME,'LabelsSemantic');
    imageFile = Iname(ind+7:end);
    fg = GetGroundTruthBB( HOMEIMAGES,HOMELABELSET,imageFile );
  else
    recs = Iname.recs;
  end
  
  % get overlaps with all ground-truths (makes sense for VOC
  % images only)
  gtbb = cat(1,fg.obj.bbox);
  os = getosmatrix_bb(boxes,gtbb);
  cats = {fg.obj.cls};
  %cats1 = cats;
  %if isfield(dataset_params,'classes')
  %  [tmp,cats] = ismember(cats,dataset_params.classes);
  %end
  
  [alpha,beta] = max(os,[],2);
  extras.maxos = alpha;
  extras.maxind = beta;
  extras.maxclass = reshape(cats(beta),size(beta));
end
