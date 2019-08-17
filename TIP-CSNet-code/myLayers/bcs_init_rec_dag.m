classdef bcs_init_rec_dag < dagnn.ElementWise
    %BCS_INIT_REC_DAG Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dims = [32 32];
    end
    
    methods
    function outputs = forward(obj, inputs, params)
        outputs{1} = bcs_initialRec_wzhshi(inputs{1},obj.dims);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = bcs_initialRec_wzhshi(inputs{1},obj.dims,derOutputs{1}); 
      derParams = {} ;
    end
    
    function obj = bcs_init_rec_dag(varargin)
      obj.load(varargin) ;
    end
    
    
    end        
end

