psf = generate_ft_gauss(24);
%load_blur;
%load_raman;
dataset_size = 200;

tic
sum_mae = 0;
cumulative_ae = 0;
baseline_mae = 0;
for ind = 1:dataset_size
    %disp(ind);
    blur = blur_spec_valid{ind,:};
    raman = raman_spec_valid{ind,:};
    b_param = table2array(blur_params(ind,1)); % first column
    psf = generate_ft_gauss(b_param);
    decon = deconvlucy(blur,psf,5);
    sum_mae = sum_mae + mae(raman,decon);
    baseline_mae = baseline_mae + mae(raman,blur)
    cumulative_ae = cumulative_ae + sum_ae(raman,decon);
end
toc

% 0.0170 at 15 iter
sum_mae/dataset_size
cumulative_ae/(dataset_size*1000)

baseline_mae/dataset_size;
