import commpy.channelcoding.convcode as cc
import numpy as np
import matplotlib.pyplot as plt
import csv

def snr_db2sigma(train_snr):
    block_len    = 100
    train_snr_Es = train_snr + 10*np.log10(float(block_len)/float(2*block_len))
    sigma_snr    = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    return sigma_snr

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_channel_low', type=float, default=0.0)
    parser.add_argument('-train_channel_high', type=float, default=8.0)

    parser.add_argument('-num_block', type=int, default=5000)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-test_ratio',  type=int, default=10)
    
    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])


    args = parser.parse_args()
    print args

    print '[ID]', args.id
    return args

    
def channel(args):
    print ('training with noise snr db', args.train_channel_low, args.train_channel_high)
    noise_sigma_low =  snr_db2sigma(args.train_channel_low) # 0dB
    noise_sigma_high =  snr_db2sigma(args.train_channel_high) # 0dB
    x=np.genfromtxt("./X_conv_train.csv")
    print ('training with noise snr db', noise_sigma_low, noise_sigma_high)
    noise_sigma =  np.random.uniform(low=noise_sigma_low,high=noise_sigma_high,size=np.shape(x))
    xx=x.reshape((args.num_block, args.block_len, 2))
    print(np.shape(xx))
    #print(np.shape(X_conv_test))
    #print(np.shape(X_conv_val))
    #print(np.shape(X_train))
    #print(np.shape(X_test))
    #print(np.shape(X_val))
    
    x=x+ noise_sigma*np.random.standard_normal(size=np.shape(x))
    with open("./Y_conv_train.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in x:
            writer.writerow([val])
    

if __name__ == '__main__':

    args = get_args()

    #train(args)
    channel(args)
