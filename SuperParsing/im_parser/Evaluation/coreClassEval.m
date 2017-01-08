function coreClassEval(HOMETESTDATA, testFileList, testParams)

fold = 'Masked R400C1111 BConst.5.1 CMIT3 CMCM6 S100 S0.00 IS0.000 Pmet IPcon SL0.00fx Seg WbS0';%SVMVal
foldNoMask = 'R400C1111 BConst.5.1 CMIT3 CMCM6 S100 S0.00 IS0.000 Pmet IPcon SL0.00fx Seg WbS0';%SVMVal
mrfFold ='MRF-SP-WithMask';
HOMETESTSET = fullfile(HOMETESTDATA,testParams.TestString);

loadFolder = fullfile(HOMETESTSET,mrfFold,'LabelsSemantic',fold);
loadFolderNoMask = fullfile(HOMETESTSET,mrfFold,'LabelsSemantic',foldNoMask);

rates = [];
for i = 1:length(testFileList)
    [fold base ext] = fileparts(testFileList{i});
    load(fullfile(loadFolder,fold,[base '.mat']));
    if(isempty(rates))
        numL = length(labelList)-1;
        rates = zeros(numL,2);
    end
    [a b] = UniqueAndCounts(L);
    b(a>numL) = [];a(a>numL) = [];
    if(isempty(b))
        load(fullfile(loadFolderNoMask,fold,[base '.mat']));
        [a b] = UniqueAndCounts(L);
        b(a>numL) = [];a(a>numL) = [];
        fprintf('No Mask ');
    end
    [foo ndx] = max(b);
    lndx = a(ndx);
    correct = 0;
    if(~isempty(lndx) && strcmp(labelList{lndx},fold))
        correct = 1;
    end
    rates(lndx,1) = rates(lndx,1)+correct;
    rates(lndx,2) = rates(lndx,2)+1;
    fprintf('%d %s-%s\n',correct,labelList{lndx},fold);
end

sum(rates(:,1))./sum(rates(:,2))
mean(rates(:,1)./rates(:,2))