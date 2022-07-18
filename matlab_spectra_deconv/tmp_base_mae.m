base_mae=0;
cum_ae =0;
load_raman;
load_blur;
dataset_size=20000;
tic
for ind = 1:dataset_size
    blur = blur_spec_valid{ind,:};
    raman = raman_spec_valid{ind,:};
    base_mae = base_mae + mae(raman,blur);
    cum_ae = cum_ae + sum_ae(raman,blur);
end

base_mae/dataset_size
cum_ae/(1000*dataset_size)
toc