psf = generate_ft_gauss(24);
%load_blur;
%load_raman;
dataset_size = 500;

tic
sum_mae = 0;
cumulative_ae = 0;
parfor ind = 1:dataset_size
    %disp(ind);
    blur = blur_spec_valid{ind,:};
    raman = raman_spec_valid{ind,:};
    decon = deconvblind(blur,psf);
    sum_mae = sum_mae + mae(raman,decon);
    cumulative_ae = cumulative_ae + sum_ae(raman,decon);
end
toc

% 0.0170 at 15 iter
sum_mae/dataset_size
cumulative_ae/(dataset_size*1000)


