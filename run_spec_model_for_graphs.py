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




min_loss=10
min_str=""
mod="/home/bhennelly/Documents/Rob-Roy/Vector_Raman_Spectra/trained_model/500k_50k/cae_9-skip/10-dataset/1.0-sigma_1.0-std/checkpoint10.pth.tar"
# used by models and blind-deconv.
dataset_val = raman_dataset('data', '4dsigma1new_0_24std_Raman_spectrums_valid.csv',
                                '4dsigma1new_0_24std_BLUR_spectrums_valid.csv')
                           
if True:
    model = CAE.CAE_9(data_len=1000, kernel_size=8, is_skip=True)

    model.cuda()
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=0)
#    checkpoint_path = os.path.join(model_save_dir, 'checkpoint' + '10' + '.pth.tar')
    checkpoint_path = mod
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    val_loss = utils.AverageMeter()  # validation loss
    full_results = None
    with torch.no_grad():
        for val_step, inputs in enumerate(tqdm(val_loader)):
            results = []
            raman = inputs[0].float().cuda()
            blur = inputs[1].float().cuda()
            outputs = model(blur)
            results.append((outputs.cpu()).numpy())
            loss_valid = loss_function(outputs, raman)
            #ind_loss = indiv_loss_function(outputs, raman)
            #pdb.set_trace()
            val_loss.update(loss_valid.item(), raman.size(0))
            results = np.array(results)
            results = results.reshape(results.shape[1], results.shape[2])
            if isinstance(full_results,type(None)):
                full_results = results.copy()
            else:
                full_results = np.concatenate((full_results,results))
        print("Exit for loop of valid")
        print('----validation----')
        a,b = 4,'d'
        pd.DataFrame(full_results).to_csv('./data/' + str(a) + b + 'test_new_Raman_spectrums_results.csv')
        #import pdb
        #pdb.set_trace()
    #     print(np.size(results))
    #     results = np.array(results)
    #     results = results.reshape(results.shape[1], results.shape[2])
    #     print(np.size(results))
    #     pd.DataFrame(results).to_csv('./data/' + str(a) + b + 'new_Raman_spectrums_results.csv')
    # print('----validation----')
    # print_string = 'loss: {loss:.5f}'.format(loss=val_loss.avg)
    # print(print_string)


