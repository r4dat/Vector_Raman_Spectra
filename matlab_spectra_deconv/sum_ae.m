function f = sum_ae(a,b)
%MAE Summary of this function goes here
%   Detailed explanation goes here
f = sum(sqrt((a-b).^2),'all');
end

