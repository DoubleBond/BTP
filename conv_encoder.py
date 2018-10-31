import numpy as np
import matplotlib.pyplot as plt
import csv
import commpy.channelcoding.convcode as cc

def conv_enc(X_train_raw, args):
    num_block = X_train_raw.shape[0]
    block_len = X_train_raw.shape[1]
    x_code    = []

    generator_matrix = np.array([[args.enc1, args.enc2]])
    M = np.array([args.M]) # Number of delay elements in the convolutional encoder
    trellis = cc.Trellis(M, generator_matrix,feedback=args.feedback)# Create trellis data structure

    for idx in range(num_block):
        xx = cc.conv_encode(X_train_raw[idx, :, 0], trellis)
        xx = xx[2*int(M):]
        xx = xx.reshape((block_len, 2))

        x_code.append(xx)

    return np.array(x_code)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_block', type=int, default=5000)
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-test_ratio',  type=int, default=10)
    parser.add_argument('-batch_size',  type=int, default=10)
    
    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")

    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    args = parser.parse_args()
    print (args)

    print ('[ID]', args.id)
    return args

def train(args):

    X_train_raw = np.random.randint(0,2,args.block_len * args.num_block)
    X_test_raw  = np.random.randint(0,2,args.block_len * args.num_block/args.test_ratio)
    X_val_raw  = np.random.randint(0,2,args.block_len * args.num_block/args.test_ratio)
    
    X_train = X_train_raw.reshape((args.num_block, args.block_len, 1))
    X_test  = X_test_raw.reshape((args.num_block/args.test_ratio, args.block_len, 1))
    X_val  = X_val_raw.reshape((args.num_block/args.test_ratio, args.block_len, 1))

    X_conv_train = conv_enc(X_train, args)
    X_conv_test  = conv_enc(X_test, args)
    X_conv_val  = conv_enc(X_val, args)

    print(np.shape(X_conv_train))
    print(np.shape(X_conv_test))
    print(np.shape(X_conv_val))
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(X_val))
    ##np.savetxt("./X_train.csv",X_train)
    ##np.savetxt("./X_val.csv",X_val)
    ##np.savetxt("./X_test.csv",X_test)
    # np.savetxt("./X_conv_train.csv",X_conv_train)
    # np.savetxt("./X_conv_val.csv",X_conv_val)
    # np.savetxt("./X_conv_test.csv",X_conv_test)
    with open("./X_train.csv", "w") as output:
        writer = csv.writer(output)
        for val in X_train:
            writer.writerow([val])

    with open("./X_test.csv", "w") as output:
        writer = csv.writer(output)
        for val in X_test:
            writer.writerow([val])      
    
    with open("./X_val.csv", "w") as output:
        writer = csv.writer(output)
        for val in X_val:
            writer.writerow([val])

    with open("./X_conv_val.csv", "w") as output:
        writer = csv.writer(output)
        for val in X_conv_val:
            writer.writerow([val])      
    

    with open("./X_conv_train.csv", "w") as output:
        writer = csv.writer(output)
        for val in X_conv_train:
            writer.writerow([val])

    with open("./X_conv_test.csv", "w") as output:
        writer = csv.writer(output)
        for val in X_conv_test:
            writer.writerow([val])      
    

if __name__ == '__main__':

    args = get_args()
    train(args)
