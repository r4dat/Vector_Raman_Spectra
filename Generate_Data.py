#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Global variables for number of data points and wavenumber axis
min_wavenumber = 0.1
max_wavenumber = 2000
n_points = 1000
step = (max_wavenumber-min_wavenumber)/(n_points)
wavenumber_axis = np.arange(min_wavenumber, max_wavenumber, step)
nu = np.linspace(0,1,n_points)

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

def generate_chi3(params):
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

    return chi3/np.max(np.abs(chi3))



#Define functions for generating nrb
def sigmoid(x,c,b):
    return 1/(1+np.exp(-(x-c)*b))


def generate_nrb():
    """
    Produces a normalized shape for the NRB
    outputs
        NRB: (n_points,)
    """
    nu = np.linspace(0,1,n_points)
    bs = np.random.normal(10/max_wavenumber,5/max_wavenumber,2)
    c1 = np.random.normal(0.2*max_wavenumber,0.3*max_wavenumber)
    c2 = np.random.normal(0.7*max_wavenumber,.3*max_wavenumber)
    cs = np.r_[c1,c2]
    sig1 = sigmoid(wavenumber_axis, cs[0], bs[0])
    sig2 = sigmoid(wavenumber_axis, cs[1], -bs[1])
    nrb  = sig1*sig2

#     plt.figure()
#     plt.plot(np.abs(nrb))
#     plt.grid()
#     plt.show()
    return nrb


def CONV(SPEC,sigma):
    from scipy.fft import fft, ifft, fftfreq, fftshift
    n=n_points
    t = np.arange(0,n)
    T=0.1e-9;
    FT_spec = fftshift(fft(fftshift(SPEC)))
    freq = fftshift(fftfreq(t.shape[-1],T))

    FT_gauss = np.exp(-0.5*(freq**2)/(sigma**2))

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

#Define functions for generating bCARS spectrum
def generate_bCARS(min_features, max_features, min_width, max_width, sigma,std):
    """
    Produces a cars spectrum.
    It outputs the normalized cars and the corresponding imaginary part.
    Outputs
        cars: (n_points,)
        chi3.imag: (n_points,)
    """
    chi3 = generate_chi3(random_parameters_for_chi3(min_features,max_features,min_width,max_width))*np.random.uniform(0.3,1) #add weight between .3 and 1
#    nrb = generate_nrb() #nrb will have valeus between 0 and 1
#    noise = np.random.randn(n_points)*np.random.uniform(0.0005,0.003)
#    bcars = ((np.abs(chi3+nrb)**2)/2+noise)
#     print(sigma)
#     print(std)
#     pause = input("Pause for key")
    bcars = chi3.imag
    tmp = CONV(chi3.imag,sigma)

    # gaussian blur the noised bcars.
    bcars= add_gaussian_white_noise(tmp,std)
    # plt.figure()
    # plt.plot(bcars)
    # plt.plot(chi3.imag)
    # plt.grid()
    # plt.show()
    return bcars, chi3.imag

def generate_batch(min_features,max_features,min_width,max_width,size = 10000,sigma_val=3e8,std_val=125e-6):
    BCARS = np.empty((size,n_points))
    RAMAN = np.empty((size,n_points))

    for i in range(size):
        BCARS[i,:], RAMAN[i,:] = generate_bCARS(min_features, max_features, min_width, max_width,sigma=sigma_val,std=std_val)
    return BCARS, RAMAN
#generate_batch(10)

def generate_all_data(min_features,max_features,min_width,max_width,N_train,N_valid):
    BCARS_train, RAMAN_train = generate_batch(min_features,max_features,min_width,max_width,N_train) # generate bactch for training
    BCARS_valid, RAMAN_valid = generate_batch(min_features,max_features,min_width,max_width,N_valid) # generate bactch for validation
    return BCARS_train, RAMAN_train, BCARS_valid, RAMAN_valid

def generate_datasets_(dataset_number,N):
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
    else:
        a=3
        b='c'
    (min_features,max_features,min_width,max_width) = key_parameters(a,b)
    BCARS, RAMAN = generate_batch(min_features,max_features,min_width,max_width,N,sigma,std) # generate bactch for training
    return BCARS, RAMAN

def generate_datasets_for_Paper_1(dataset_number,N):
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
    else:
        a=3
        b='c'
    (min_features,max_features,min_width,max_width) = key_parameters(a,b)
    BCARS, RAMAN = generate_batch(min_features,max_features,min_width,max_width,N,sigma,std) # generate bactch for training
    X = np.empty((N, n_points,1))
    y = np.empty((N,n_points))

    for i in range(N):
        X[i,:,0] = BCARS[i,:]
        y[i,:] = RAMAN[i,:]
    return X, y

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
    else:
        a=3
        b='c'
    (min_features,max_features,min_width,max_width) = key_parameters(a,b)
    BCARS, RAMAN = generate_batch(min_features,max_features,min_width,max_width,N,sigma,std) # generate bactch for training
    return BCARS, RAMAN
#    X = np.empty((N, n_points,1))
#    y = np.empty((N,n_points))

#    for i in range(N):
#        X[i,:,0] = BCARS[i,:]
#        y[i,:] = RAMAN[i,:]
#    return X, y
def generate_one_spectrum_Paper_1(dataset_number):
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
    else:
        a=3
        b='c'
    (min_features,max_features,min_width,max_width) = key_parameters(a,b)
    BCARS, RAMAN = generate_bCARS(min_features, max_features, min_width, max_width)  # generate bactch for training
    return BCARS, RAMAN


#save batch to memory for training and validation - this is optional if we want to make sure the same data was used to train different methods
#it is obviously MUCH faster to generate data on the fly and not read to/write from RzOM

def generate_and_save_data(N_train,N_valid,fname='./data/',a=1,b='a',sigma_val=3e8,std_val=125e-6):

    (min_features,max_features,min_width,max_width) = key_parameters(a,b)

    print('min_features=',min_features,'max_features=',max_features,'min_width=',min_width,'max_width=',max_width)

    BCARS_train, RAMAN_train, BCARS_valid, RAMAN_valid = generate_all_data(min_features,max_features,min_width,max_width,N_train,N_valid)

    print(np.isinf(BCARS_train).any())
    print(np.isinf(RAMAN_train).any())
    print(np.isnan(BCARS_train).any())
    print(np.isnan(RAMAN_train).any())
    print(np.isinf(BCARS_valid).any())
    print(np.isinf(RAMAN_valid).any())
    print(np.isnan(BCARS_valid).any())
    print(np.isnan(RAMAN_valid).any())

    pd.DataFrame(RAMAN_valid).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'Raman_spectrums_valid.csv')
    pd.DataFrame(BCARS_valid).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'CARS_spectrums_valid.csv')
    pd.DataFrame(RAMAN_train).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'Raman_spectrums_train.csv')
    pd.DataFrame(BCARS_train).to_csv(fname+str(a)+b+'sigma'+str(sigma_val)+'std'+str(std_val)+'CARS_spectrums_train.csv')

    return BCARS_train, RAMAN_train, BCARS_valid, RAMAN_valid

def load_data(name1,name2):
    # load training set
    RAMAN_train = pd.read_csv(name1)
    BCARS_train = pd.read_csv(name2)

    plt.figure()
    plt.plot(RAMAN_train[2:4])
    plt.show()

    # load validation set
    RAMAN_valid = pd.read_csv('./data/3bRaman_spectrums_valid.csv')
    BCARS_valid = pd.read_csv('./data/3bCARS_spectrums_valid.csv')

    RAMAN_train = RAMAN_train.values[:,1:]
    BCARS_train = BCARS_train.values[:,1:]
    RAMAN_valid = RAMAN_valid.values[:,1:]
    BCARS_valid = BCARS_valid.values[:,1:]


    return BCARS_train, RAMAN_train, BCARS_valid, RAMAN_valid

if __name__=='__main__':
    generate_and_save_data(N_train=1000,N_valid=1000,fname='./data/',a=1,b='a',sigma_val=3e8,std_val=125e-6) #1
    generate_and_save_data(N_train=1000,N_valid=1000,fname='./data/',a=3,b='c',sigma_val=3e8,std_val=125e-6) #9