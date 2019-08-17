%% Experiment with the cnn_mnist_fc_bnorm
clc
close all

[net_bn, info_bn] = cnn_CSNetPlus_dag(...
  'expDir', './results');
