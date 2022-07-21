import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import scipy.signal
from tensorboardX import SummaryWriter
import time
import pdb
import argparse
from skimage import restoration
from scipy.signal import fftconvolve
from scipy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft

# self defined modules
from models import CAE
import utils


def loss_function(recon_x, x):  # hidden_neurons_batch: [samples, number of neurons]
    # BCE = F.mse_loss(recon_x.view(-1, 1000), x.view(-1, 1000))
    BCE = F.l1_loss(recon_x.view(-1, 1000), x.view(-1, 1000))

    return BCE.cuda()

def indiv_loss_function(recon_x, x):  # hidden_neurons_batch: [samples, number of neurons]
    # BCE = F.mse_loss(recon_x.view(-1, 1000), x.view(-1, 1000))
    BCE = F.l1_loss(recon_x.view(-1, 1000), x.view(-1, 1000),reduction='none')

    return BCE.cuda()

from Generate_Data import *


class raman_dataset_fast(Dataset):
    def __init__(self, dataset, size):
        self.blur_data, self.raman_data = generate_datasets(dataset, size, sigma=args.sigma, std=args.std)

    def __len__(self):
        return len(self.raman_data)

    def __getitem__(self, idx):
        raman_data = self.raman_data[idx]
        blur_data = self.blur_data[idx]
        return raman_data, blur_data


class raman_dataset(Dataset):
    def __init__(self, file_path, raman_file, blur_file):
        self.raman_data = pd.read_csv(os.path.join(file_path, raman_file)).iloc[:, 1:]
        self.blur_data = pd.read_csv(os.path.join(file_path, blur_file)).iloc[:, 1:]

    def __len__(self):
        return len(self.raman_data)

    def __getitem__(self, idx):
        raman_data = self.raman_data.values[idx]
        blur_data = self.blur_data.values[idx]
        return raman_data, blur_data



listmodels = list()

for (dirpath, dirnames, filenames) in os.walk('/home/bhennelly/Documents/Rob-Roy/Vector_Raman_Spectra/trained_model'):
    listmodels += [os.path.join(dirpath, file) for file in filenames]

min_loss=10
min_str=""
mod=""
# used by models and blind-deconv.
dataset_val = raman_dataset('data', '4dsigma1new_0_24std_Raman_spectrums_valid.csv',
                                '4dsigma1new_0_24std_BLUR_spectrums_valid.csv')
for mod in listmodels:
#if True:
    if 'cae_9' in mod:
        model = CAE.CAE_9(data_len=1000, kernel_size=8, is_skip=True)
    elif 'cae_4-skip' in mod:
        model = CAE.CAE_4(data_len=1000, kernel_size=8, is_skip=True)
    elif 'cae_4_drop' in mod:
        model = CAE.CAE_4_DROP(data_len=1000, kernel_size=8, is_skip=True)

    model.cuda()
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=0)
#    checkpoint_path = os.path.join(model_save_dir, 'checkpoint' + '10' + '.pth.tar')
    checkpoint_path = mod
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    val_loss = utils.AverageMeter()  # validation loss
    with torch.no_grad():
        results = []
        for val_step, inputs in enumerate(tqdm(val_loader)):
            raman = inputs[0].float().cuda()
            blur = inputs[1].float().cuda()
            outputs = model(blur)
            results.append((outputs.cpu()).numpy())
            loss_valid = loss_function(outputs, raman)
            #ind_loss = indiv_loss_function(outputs, raman)
            #pdb.set_trace()
            val_loss.update(loss_valid.item(), raman.size(0))
        print("Exit for loop of valid")
        print('----validation----')
        print('\n'+mod+'\n')
        print_string = 'loss: {loss:.5f}'.format(loss=val_loss.avg)
        print(print_string)
        if(val_loss.avg)<min_loss:
            min_loss = val_loss.avg
            min_str = mod
        # import pdb
        # pdb.set_trace()
    #     print(np.size(results))
    #     results = np.array(results)
    #     results = results.reshape(results.shape[1], results.shape[2])
    #     print(np.size(results))
    #     pd.DataFrame(results).to_csv('./data/' + str(a) + b + 'new_Raman_spectrums_results.csv')
    # print('----validation----')
    # print_string = 'loss: {loss:.5f}'.format(loss=val_loss.avg)
    # print(print_string)

print(min_loss)
print(mod)

base_mae_value = 0
# run blind deconv

def get_ft_gaussian(sigma_cm): # 1000 points, range 2000-0.01
    #Sigma is the standard deviation of the impulse response (assumed Gaussian) should be defined in units of cm-1
    n=1000
    t = np.arange(0,n)
    #T=0.1e-9;
    Range = 2000-0.1
    T=Range/n
    freq = fftshift(fftfreq(t.shape[-1],T))
    FT_gauss = np.exp(-2*(np.pi**2)*(freq**2)*(sigma_cm**2))
    return FT_gauss

def blind_richardson_lucy(spectra, psf, num_iter=5):    
    sp_deconv = np.full(spectra.shape, np.median(spectra), dtype='float') # initialize the deconvolved spectra
    orig_spectra = spectra
    orig_psf = psf
    for i in range(num_iter):
        psf_mirror = np.flip(psf)
        conv = fftconvolve(sp_deconv, psf, mode='same')
        with np.errstate(all='raise'):
            try:
                relative_blur = spectra / conv
            except FloatingPointError:
                tmp = blind_richardson_lucy(orig_spectra,orig_psf,num_iter=(i-1))
                sp_deconv=tmp
                return tmp
        sp_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')
        sp_deconv_mirror = np.flip(sp_deconv)
        psf *= fftconvolve(relative_blur, sp_deconv_mirror, mode='same')  
        if (np.max(sp_deconv)==0 and i>1) or (np.max(np.abs(sp_deconv))>1.1) or (np.isnan(sp_deconv).any()):
            tmp = blind_richardson_lucy(orig_spectra,orig_psf,num_iter=(i-1))
            sp_deconv=tmp
            return tmp      
    return sp_deconv
    
def np_mae(actual,predicted):
    return np.mean(np.abs(actual - predicted))

if True:
    for index in range(0,len(dataset_val)):
        raman, blur = dataset_val[index]
        base_mae_value = base_mae_value + np_mae(raman,blur)


print("Baseline MAE:")
print(str((base_mae_value/len(dataset_val))))


blind_rl_mae = 0
# RUN blind-richardson-lucy
if False:
    psf = get_ft_gaussian(24)
    for index in range(0,len(dataset_val)):
        raman, blur = dataset_val[index]
        decon = blind_richardson_lucy(blur, psf, 10)
        if np.isnan(np_mae(raman,decon)):
            print("data_row is: "+str(index))
        blind_rl_mae = blind_rl_mae + np_mae(raman,decon)
        if False:
            from matplotlib import pyplot as plt
            plt.plot(raman,label="Raman")
            plt.plot(blur,label="Blur")
            plt.plot(decon,label="Decon")
            plt.title("Data index is: "+str(index))
            plt.legend()
            plt.show()
        

print("RL Decon MAE:")
print(str((blind_rl_mae/len(dataset_val))))
