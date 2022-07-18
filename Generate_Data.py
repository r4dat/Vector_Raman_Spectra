#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft
import random

#Global variables for number of data points and wavenumber axis
min_wavenumber = 0.1
max_wavenumber = 2000
Range = max_wavenumber-min_wavenumber
n_points = 1000
step = (max_wavenumber-min_wavenumber)/(n_points)
wavenumber_axis = np.arange(min_wavenumber, max_wavenumber, step)
nu = np.linspace(0,1,n_points)

def sigma_func():
    return round(np.random.uniform(0,24),2)
    
def snr_func():
    return round(random.choice([np.random.uniform(20,100),np.random.uniform(100,1000)]),2)

#Global variables for benchmarking (number of peaks and FWHM width of peaks)
#CASE 1 - 2-10cm-1 width
#CASE 2 - 2-25cm-1 width
#CASE 3 - 2-75cm-1 width

#CASE A - 1-15 peaks
#CASE B - 15-30 peaks
#CASE C - 30-50 peaks

#set case
def key_parameters(a=3,b='c'):
    if a == 1 and b == 'a' :
        #CASE 1_A
        min_features = 1
        max_features = 15
        min_width = 2
        max_width = 10
    elif a == 1 and b == 'b' :
        #CASE 1_B
        min_features = 15
        max_features = 30
        min_width = 2
        max_width = 10
    elif a == 1 and b == 'c' :
        #CASE 1_A
        min_features = 30
        max_features = 50
        min_width = 2
        max_width = 10
    elif a == 2 and b == 'a' :
        #CASE 1_A
        min_features = 1
        max_features = 15
        min_width = 2
        max_width = 25
    elif a == 2 and b == 'b' :
        #CASE 1_B
        min_features = 15
        max_features = 30
        min_width = 2
        max_width = 25
    elif a == 2 and b == 'c' :
        #CASE 1_A
        min_features = 30
        max_features = 50
        min_width = 2
        max_width = 25
    elif a == 3 and b == 'a' :
        #CASE 1_A
        min_features = 1
        max_features = 15
        min_width = 2
        max_width = 75
    elif a == 3 and b == 'b' :
        #CASE 1_B
        min_features = 15
        max_features = 30
        min_width = 2
        max_width = 75
    elif a == 3 and b == 'c' :
        #CASE 1_A
        min_features = 30
        max_features = 50
        min_width = 2
        max_width = 75
    elif a == 4 and b == 'd' :
        #CASE 1_A
        min_features = 1 
        max_features = 50 
        min_width = 0.5
        max_width = np.random.randint(15, 75)
    else:
        print('Case not defined correctly')
    return (min_features,max_features,min_width,max_width)


#Define functions for generating suseptibility
def random_parameters_for_chi3(min_features,max_features,min_width,max_width):
    """
    generates a random spectrum, without NRB.
    output:
        params =  matrix of parameters. each row corresponds to the [amplitude, resonance, linewidth] of each generated feature (n_lor,3)
    """
    n_lor = np.random.randint(min_features,max_features+1) #the +1 was edited from bug in Paper 1.
    a = np.random.uniform(0,1,n_lor) #these will be the amplitudes of the various lorenzian function (A) and will vary between 0 and 1
    w = np.random.uniform(min_wavenumber+300,max_wavenumber-300,n_lor) #these will be the resonance wavenumber poisitons
    g = np.random.uniform(min_width,max_width, n_lor) # and tehse are the width

    params = np.c_[a,w,g]
#    print(params)
    return params

def generate_Raman(params):
    """
    buiilds the normalized chi3 complex vector
    inputs:
        params: (n_lor, 3)
    outputs
        chi3: complex, (n_points, )
    """

    chi3 = np.sum(params[:,0]/(-wavenumber_axis[:,np.newaxis]+params[:,1]-1j*params[:,2]),axis = 1)

#     plt.figure()
#     plt.plot(np.real(chi3))
#     plt.grid()
#     plt.show()

#     plt.figure()
#     plt.plot(np.imag(chi3))
#     plt.grid()
#     plt.show()

#     plt.figure()
#     plt.plot(np.abs(chi3))
#     plt.grid()
#     plt.show()

#     plt.figure()
#     plt.plot(np.angle(chi3))
#     plt.grid()
#     plt.show()
    spec = chi3.imag
    spec = spec - np.min(spec)
    spec = spec/np.max(spec)
    
    return spec



def CONV(SPEC,sigma_cm):
    #Sigma is the standard deviation of the impulse response (assumed Gaussian) should be defined in units of cm-1
    n=n_points
    t = np.arange(0,n)
    #T=0.1e-9;
    T=Range/n
    FT_spec = fftshift(fft(fftshift(SPEC)))
    freq = fftshift(fftfreq(t.shape[-1],T))
    FT_gauss = np.exp(-2*(np.pi**2)*(freq**2)*(sigma_cm**2))
    
    #plt.plot(wavenumber_axis,np.abs(fftshift(fft(fftshift(FT_gauss)))))
    #test_gauss = 2*(1/sigma_cm)*(1/((2*np.pi)**0.5))*np.exp(-((wavenumber_axis-Range/2)**2)/(2*sigma_cm**2))
    #plt.plot(wavenumber_axis,test_gauss,'-r')
    #plt.show()
    
    #FT_test_gauss = fftshift(fft(fftshift(test_gauss)))
    #plt.plot(freq,FT_test_gauss,'-r')
    #plt.plot(freq,FT_gauss)
    #plt.show()


    
    
    FT_blur = FT_spec*FT_gauss
    blur = fftshift(ifft(fftshift(FT_blur)))
    blur=np.abs(blur)
    return(np.abs(blur))

def add_gaussian_white_noise(SPEC,std=0.01):
    mean = 0
    samples = np.random.normal(mean, std, size=n_points)
    out = SPEC+samples
    out = out + (np.ndarray.min(out))*-1 # force >= 0.
    return out

# add poisson point noise to spectra
def poisson_point_noise(BLUR,RAMAN,SNR):
    #SNR is defined as E[P_sig]/E[P_noise] where E is expected value or mean. We define as E[max[P_sig]]/E[noise at max[P_sig]]
    #For a shot noised signal we take the maximum value to be X. For this value the std_dev of the Poisson Noise Distn is root(X)
    #Therefore SNR is defined as X/root(X) where X is the max value
    #Selecting an SNR value SNR: the max irradiance MAX must be given by: SNR=root(MAX), => MAX=SNR^2
    #Therefore for a normalised signal, the signal should be scaled by 1/SNR^2
    scale = SNR**2
    IRRADIANCE = BLUR * scale
    RAMAN = RAMAN * scale
    NOISEY = np.random.poisson(lam=IRRADIANCE)
    #now normalise identically
    NOISEY = NOISEY - np.min(RAMAN)
    RAMAN = RAMAN - np.min(RAMAN) # original all noisy
    NOISEY = NOISEY/np.max(RAMAN)
    RAMAN = RAMAN/np.max(RAMAN)
    return NOISEY,RAMAN
	

#Define functions for generating bCARS spectrum
def generate_spec(min_features, max_features, min_width, max_width, sigma,SNR,std):
    """
    Produces a cars spectrum.
    It outputs the normalized cars and the corresponding imaginary part.
    Outputs
        cars: (n_points,)
        chi3.imag: (n_points,)
    """
    Raman = generate_Raman(random_parameters_for_chi3(min_features,max_features,min_width,max_width))*np.random.uniform(0.3,1) #add weight between .3 and 1
#    nrb = generate_nrb() #nrb will have valeus between 0 and 1
#    noise = np.random.randn(n_points)*np.random.uniform(0.0005,0.003)
#    bcars = ((np.abs(chi3+nrb)**2)/2+noise)
#     print(sigma)
#     print(std)
#     pause = input("Pause for key")
    sigma = sigma_func() # see globals at top - rounded uniform(0,24)
    Raman_blur = CONV(Raman,sigma)

    # gaussian blur the noised bcars.
   # Raman= add_gaussian_white_noise(Raman,std)
    # plt.figure()
    # plt.plot(bcars)
    # plt.plot(chi3.imag)
    # plt.grid()
    # plt.show()
    SNR = snr_func() # see globals at top, rounded 50/50 uniform(20,100), uniform(100,1000)
    Raman_blur, Raman = poisson_point_noise(Raman_blur, Raman,SNR)
    return Raman_blur, Raman

def generate_batch(min_features,max_features,min_width,max_width,size = 10000,sigma_val=3e8,SNR=100,std_val=125e-6):
    BLUR = np.empty((size,n_points))
    RAMAN = np.empty((size,n_points))

    for i in range(size):
        BLUR[i,:], RAMAN[i,:] = generate_spec(min_features, max_features, min_width, max_width,sigma_val,SNR,std_val)
    return BLUR, RAMAN
#generate_batch(10)

def generate_all_data(min_features,max_features,min_width,max_width,N_train,N_valid):
    BLUR_train, RAMAN_train = generate_batch(min_features,max_features,min_width,max_width,N_train) # generate bactch for training
    BLUR_valid, RAMAN_valid = generate_batch(min_features,max_features,min_width,max_width,N_valid) # generate bactch for validation
    return BLUR_train, RAMAN_train, BLUR_valid, RAMAN_valid

def generate_datasets(dataset_number,N,sigma,SNR,std):
    if dataset_number == 1:
        a=1
        b='a'
    elif dataset_number == 2:
        a=1
        b='b'
    elif dataset_number == 3:
        a=1
        b='c'
    elif dataset_number == 4:
        a=2
        b='a'
    elif dataset_number == 5:
        a=2
        b='b'
    elif dataset_number == 6:
        a=2
        b='c'
    elif dataset_number == 7:
        a=3
        b='a'
    elif dataset_number == 8:
        a=3
        b='b'
    elif dataset_number == 9:
        a=3
        b='c'
    elif dataset_number == 10:
        a= 4
        b='d'
    
    (min_features,max_features,min_width,max_width) = key_parameters(a,b)
    BLUR, RAMAN = generate_batch(min_features,max_features,min_width,max_width,N,sigma,SNR,std) # generate bactch for training
    return BLUR, RAMAN


def generate_datasets(dataset_number,N,sigma,std):
    if dataset_number == 1:
        a=1
        b='a'
    elif dataset_number == 2:
        a=1
        b='b'
    elif dataset_number == 3:
        a=1
        b='c'
    elif dataset_number == 4:
        a=2
        b='a'
    elif dataset_number == 5:
        a=2
        b='b'
    elif dataset_number == 6:
        a=2
        b='c'
    elif dataset_number == 7:
        a=3
        b='a'
    elif dataset_number == 8:
        a=3
        b='b'
    elif dataset_number == 9:
        a=3
        b='c'
    elif dataset_number == 10:
        a= 4
        b='d'
    (min_features,max_features,min_width,max_width) = key_parameters(a,b)
    BCARS, RAMAN = generate_batch(min_features,max_features,min_width,max_width,N,sigma,std) # generate bactch for training
    return BCARS, RAMAN
#    X = np.empty((N, n_points,1))
#    y = np.empty((N,n_points))

#    for i in range(N):
#        X[i,:,0] = BCARS[i,:]
#        y[i,:] = RAMAN[i,:]
#    return X, y


#save batch to memory for training and validation - this is optional if we want to make sure the same data was used to train different methods
#it is obviously MUCH faster to generate data on the fly and not read to/write from RzOM

def generate_and_save_data(N_train,N_valid,fname='./data/',a=1,b='a',sigma_val=3e8,std_val=125e-6):

    (min_features,max_features,min_width,max_width) = key_parameters(a,b)

    print('min_features=',min_features,'max_features=',max_features,'min_width=',min_width,'max_width=',max_width)

    BLUR_train, RAMAN_train, BLUR_valid, RAMAN_valid = generate_all_data(min_features,max_features,min_width,max_width,N_train,N_valid)

    print(np.isinf(BLUR_train).any())
    print(np.isinf(RAMAN_train).any())
    print(np.isnan(BLUR_train).any())
    print(np.isnan(RAMAN_train).any())
    print(np.isinf(BLUR_valid).any())
    print(np.isinf(RAMAN_valid).any())
    print(np.isnan(BLUR_valid).any())
    print(np.isnan(RAMAN_valid).any())

    sigma_val="1new_0_24"
    std_val="_"
    pd.DataFrame(RAMAN_valid).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'Raman_spectrums_valid.csv')
    pd.DataFrame(BLUR_valid).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'BLUR_spectrums_valid.csv')
    pd.DataFrame(RAMAN_train).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'Raman_spectrums_train.csv')
    pd.DataFrame(BLUR_train).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'BLUR_spectrums_train.csv')

    return BLUR_train, RAMAN_train, BLUR_valid, RAMAN_valid

def load_data(name1,name2):
    # load training set
    RAMAN_train = pd.read_csv(name1)
    BLUR_train = pd.read_csv(name2)

    plt.figure()
    plt.plot(RAMAN_train[2:4])
    plt.show()

    # load validation set
    RAMAN_valid = pd.read_csv('./data/3bRaman_spectrums_valid.csv')
    BLUR_valid = pd.read_csv('./data/3bCARS_spectrums_valid.csv')

    RAMAN_train = RAMAN_train.values[:,1:]
    BLUR_train = BCARS_train.values[:,1:]
    RAMAN_valid = RAMAN_valid.values[:,1:]
    BLUR_valid = BCARS_valid.values[:,1:]

    return BLUR_train, RAMAN_train, BLUR_valid, RAMAN_valid

def test():
    sigma = 12
    std=0
    length=10
    SNR=50
 
    sigma_list = [12,24,36,48]
    snr_list = [5,10,25,50,75,100]
    import random

    for n in range(length):
        s_ind = random.randint(0,len(sigma_list)-1)
        snr_ind = random.randint(0,len(snr_list)-1)
        sigma = sigma_list[s_ind]
        SNR = snr_list[snr_ind]
        A,B = generate_datasets(9,length,sigma,SNR,std)
        plt.plot(A[n,:],'-r')
        plt.plot(B[n,:],'-b') 
        title_str = 'Sigma: ' + str(sigma) + ' SNR: '+str(SNR)
        plt.title(title_str)
        plt.show() 


if __name__=='__main__':
    generate_and_save_data(N_train=20000,N_valid=20000,fname='./data/',a=4,b='d',sigma_val=2e8,std_val=125e-4) #10
