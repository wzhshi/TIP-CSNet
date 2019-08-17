function net = cnn_CSNetPlus_init_dagnn( varargin )
%CNN_DDCN_INIT Summary of this function goes here
%   Detailed explanation goes here
% opts.networkType = 'dagnn';
% opts = vl_argparse(opts,varargin);

net = dagnn.DagNN();
k=50;
d=64;
rng('default');
rng(0) ;

reluLeak = 0;
% bnormal =false;
net.meta.solver = 'Adam';
net.meta.inputSize = [96 96] ;
net.meta.trainOpts.batchSize = 64;
net.meta.trainOpts.learningRate = [logspace(-3,-3,50) logspace(-4,-4,30) logspace(-5,-5,20)];
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.adjGradClipping = false;
net.meta.derOutputs = {'pdist1',1,'pdist2',1};
%for raw
net.meta.imdbPath = './data/model_64_96_Adam/imdb.mat';

% sampling 

%%% Here the example corresponds to the sampling rato of 0.1, so there are
%%% 102 measurements for one block. For other sampling ratios, please
%%% change 102 to other value. For example, change 102 to 10 for the
%%% sampling ratio of 0.01.

block = dagnn.Conv('size',  [32 32 1 102], 'hasBias', false, ...
                   'stride', 32, 'pad', [0 0 0 0]);
lName = 'sampling';
net.addLayer(lName, block, 'input', lName, {[lName '_f_notBi']});

%initial reconstruction
block = dagnn.Conv('size',  [1 1 102 1024], 'hasBias', false, ...
                   'stride', 1, 'pad', [0 0 0 0]);
lName = 'initRecon';
net.addLayer(lName, block, 'sampling', lName, {[lName '_f']});

block = dagnn.bcs_init_rec_dag('dims',[32 32]);
lName = 'combine';
net.addLayer(lName,block,'initRecon',lName);

%deep reconstruction
block = dagnn.Conv('size',  [3 3 1 d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName1 = 'dr1';
net.addLayer(lName1, block, 'combine', lName1, {[lName1 '_f'], [lName1 '_b']});
block = dagnn.ReLU('leak',reluLeak);
net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);

for i=2:2:k
    block = dagnn.Conv('size',  [3 3 d d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
    lName2 = ['dr' num2str(i)];
    net.addLayer(lName2, block, ['dr' num2str(i-1) '_relu'], lName2, {[lName2 '_f'], [lName2 '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName2 '_relu'],  block, lName2, [lName2 '_relu']);
    
    net.addLayer(['res' num2str(i)],dagnn.Sum(),{[lName1 '_relu'],[lName2 '_relu']},['res' num2str(i)]);
    
    block = dagnn.Conv('size',  [3 3 d d], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
    lName1 = ['dr' num2str(i+1)];
    net.addLayer(lName1, block, ['res' num2str(i)], lName1, {[lName1 '_f'], [lName1 '_b']});
    block = dagnn.ReLU('leak',reluLeak);
    net.addLayer([lName1 '_relu'],  block, lName1, [lName1 '_relu']);         
end

block = dagnn.Conv('size',  [3 3 d 1], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = 'dr_pred';
net.addLayer(lName, block, ['dr' num2str(i+1) '_relu'], lName, {[lName '_f'], [lName '_b']});
    
net.addLayer('prediction',dagnn.Sum(),{'dr_pred','combine'},'prediction');

net.addLayer('pdist1',dagnn.EuclidLoss(),{'prediction','label'},'pdist1');
net.addLayer('pdist2',dagnn.EuclidLoss(),{'combine','label'},'pdist2');


net.initParams();
