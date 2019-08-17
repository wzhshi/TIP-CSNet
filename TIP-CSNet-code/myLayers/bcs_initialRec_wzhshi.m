function y = bcs_initialRec_wzhshi( x, dims, dzdy )
%BCS_INITIALREC_WZHSHI Summary of this function goes here
%   Detailed explanation goes here
sz = size(x) ;

if nargin <= 2 || isempty(dzdy)
    
  dims = horzcat(dims, 1) ; %gray  
%    dims = horzcat(dims, 3) ; %color
%     dims = horzcat(dims, size(x,4)) ;
    y = [];
    for i=1:sz(1)
        temp = [];
        for j=1:sz(2)
          temp = [temp vl_nnreshape_wzhshi(x(i,j,:,:),dims)];         
        end
        y = [y;temp];
    end
else
%     dims = horzcat(dims, 3) ; %color
    dims = horzcat(dims, 1) ; %gray
    y = x;
    for i=1:sz(1)
        for j = 1:sz(2)
            y(i,j,:,:) = vl_nnreshape_wzhshi(dzdy((i-1)*dims(1)+1 : i*dims(1),...
                (j-1)*dims(2)+1 : j*dims(2) , : , :), [1 1 dims(1)*dims(2)*dims(3)]);
                        
        end
    end
   
    
    
end



