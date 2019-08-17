clear all; close all;

run 'D:\AcademicSoftware\matconvnet-1.0-beta25\matlab\vl_setupnn.m'
addpath('./data/utilities');
ds = {'Set5' 'Set14','BSD100','Sun-Hays80','Urban100'};

SamplingRatio = 0.01; % The pretrained model corresponds to the samplingratios of 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, and 0.5.

 net = load(['./model/net-' num2str(SamplingRatio) '.mat']);
 net = dagnn.DagNN.loadobj(net.net);
  
showResult  = 0;
useGPU      = 1;
pauseTime   = 0;
if useGPU
    net.move('gpu') ;
end

for k=1
dataSet = ds{k};

folderTest = ['./TestImage/' dataSet];

ext         =  {'*.jpg','*.png','*.bmp'};
filepaths   =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folderTest,ext{i})));
end


    PSNRs_CSNet = zeros(1,length(filepaths));
    SSIMs_CSNet = zeros(1,length(filepaths));
    
for i = 1:length(filepaths)
    image = imread(fullfile(folderTest,filepaths(i).name));
    [~,nameCur,extCur] = fileparts(filepaths(i).name);
    if size(image,3)==3
    image = rgb2ycbcr(image);
    image = im2single(image(:, :, 1));

    else
        image =im2single(image);
    end
    image = modcrop(image,32); 
    input = image;
    label = image;
    
   if useGPU
        input = gpuArray(input);
    end
    net.eval({'input',input});
    output = net.vars(end-3).value;

   if useGPU
        output = gather(output);
   end

    [PSNRCur_CSNet, SSIMCur_CSNet] = Cal_PSNRSSIM(im2uint8(output),im2uint8(label),0,0);
   if showResult
        imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        title([filepaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
        pause(pauseTime)
    end 
        
    PSNRs_CSNet(i) = PSNRCur_CSNet;
    SSIMs_CSNet(i) = SSIMCur_CSNet;

end
 disp([mean(PSNRs_CSNet),mean(SSIMs_CSNet)]);
end