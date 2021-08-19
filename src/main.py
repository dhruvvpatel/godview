# training the network

import os
import sys
import time
import argparse

import torch
import torch.optim as optim
import numpy as np
import model

from dataloader import *
from helper import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 227
kGeneratedExamplesPerImage = 10     # generate 10 synthetic samples per image
transform = NormalizeToTensor()
bb_params = {}

if True:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()


'''
get the following things from the user
- total batches                         n
- learning rate                         lr
- data directory                        d
- result directory                      s 
- lambda shift for random crop          lshift
- lambda scale for random crop          lscale  
- min scale for random crop             min
- max scale for random crop             max
- manual seed                           seed
- batch size                            BS
- save frequency                        savefq
'''

parser = argparse.ArgumentParser(description='TRAINING')

parser.add_argument('-n', default=500000, type=int, help='number of total batches')
parser.add_argument('-lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('-d', type=str, help='path to data directory')
parser.add_argument('-s', type=str, help='path to results directory')
parser.add_argument('-lshift', default=5, type=float, help='lambda-shift for random crop')
parser.add_argument('-lscale', default=15, type=float, help='lambda-scale for random crop')
parser.add_argument('-min', default=-0.4, type=float, help='min-scale for random crop')
parser.add_argument('-max', default=0.4, type=float, help='max-scale for random crop')
parser.add_argument('-seed', default=44, type=int, help='manual seed')
parser.add_argument('-BS', default=50, type=int, help='batch-size')
parser.add_argument('-savefq', default=20000, type=int, help='after how many steps to save the model')


# sanity check :: parse and print all the arguments
args = parser.parse_args()
print(args)

# take what you need
bs = args.BS
lr = args.lr
savefq = args.savefq
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# bb motion model params
bb_params['lambda_shift_frac'] = args.lshift
bb_params['lambda_scale_frac'] = args.lscale
bb_params['min_scale'] = args.min
bb_params['max_scale'] = args.max

# load dataset using dataloader
alov = ALOVDataset(os.path.join(args.d, 'imagedata++/'),
                   os.path.join(args.d, 'alov300++_rectangleAnnotation_full/'),
                   transform,
                   input_size)

# dataset list
datasets = [alov]           # for now only alov

# model 
net = model.GoNet().to(device)
loss_fn = torch.nn.L1Loss().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)


if os.path.exists(args.s):
    print(f' Directory {args.s} already exists ')
else:
    os.makedirs(args.s)
    print(f' Directory {args.s} created ... ')




def make_transformed_samples(dataset, args):
    '''
    --> Given a dataset, it picks a random sample from it and returns a batch of (kGeneratedExamplesPerImage + 1) samples. The batch contains true sample from dataset and kGeneratedExamplesPerImage samples, which are created artifically with augmentation by GOTURN smooth motion model.
    '''

    idx = np.random.randint(dataset.len, size=1)[0]
    # unscaled original sample (single image and bb)
    orig_sample = dataset.get_orig_sample(idx)
    # cropped scaled sample (two frames and bb)
    true_sample, _ = dataset.get_sample(idx)
    true_tensor = transform(true_sample)

    x1_batch = torch.Tensor(kGeneratedExamplesPerImage+1, 3, input_size, input_size)
    x2_batch = torch.Tensor(kGeneratedExamplesPerImage+1, 3, input_size, input_size)
    y_batch = torch.Tensor(kGeneratedExamplesPerImage+1, 4)

    # initialize batch with the true sample
    x1_batch[0] = true_tensor['previmg']
    x2_batch[0] = true_tensor['currimg']
    y_batch[0] = true_tensor['currbb']

    scale = Rescale((input_size, input_size))

    for i in range(kGeneratedExamplesPerImage):
        sample = orig_sample
        
        # unscaled current imge crop with box
        curr_sample, opts_curr = shift_crop_training_sample(sample, bb_params)
        # unscaled previous image crop with box
        prev_sample, opts_prev = crop_sample(sample)
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)
        training_sample = {'previmg' : scaled_prev_obj['image'],
                           'currimg' : scaled_curr_obj['image'],
                           'currbb'  : scaled_curr_obj['bb']}

        sample = transform(training_sample)
        x1_batch[i+1] = sample['previmg']
        x2_batch[i+1] = sample['currimg']
        y_batch[i+1] = sample['currbb']

    return x1_batch, x2_batch, y_batch


def get_training_batch(num_running_batch, running_batch, dataset, args, bs):
    '''
    GOTURN batch formation 
    '''

    isdone = False
    N = kGeneratedExamplesPerImage+1
    train_batch = None
    x1_batch, x2_batch, y_batch = make_transformed_samples(dataset, args)
    assert(x1_batch.shape[0] == x2_batch.shape[0]  == y_batch.shape[0] == N)
    count_in = min(bs - num_running_batch, N)
    remain = N - count_in

    running_batch['previmg'][num_running_batch:num_running_batch+count_in] = x1_batch[:count_in]
    running_batch['currimg'][num_running_batch:num_running_batch+count_in] = x2_batch[:count_in]
    running_batch['currbb'][num_running_batch:num_running_batch+count_in] = y_batch[:count_in]

    num_running_batch += count_in

    if remain > 0:
        isdone = True
        train_batch = running_batch.copy()
        running_batch['previmg'][:remain] = x1_batch[-remain:]
        running_batch['currimg'][:remain] = x2_batch[-remain:]
        running_batch['currbb'][:remain] = y_batch[-remain:]
        num_running_batch = remain
    
    return running_batch, train_batch, isdone, num_running_batch



# LET'S TRAIN NOW ... ...

since = time.time()
curr_loss = 0
itr = 0
num_running_batch = 0
running_batch = {'previmg' : torch.Tensor(bs, 3, input_size, input_size),
                 'currimg' : torch.Tensor(bs, 3, input_size, input_size),
                 'currbb'  : torch.Tensor(bs, 4)}

if not os.path.isdir(args.s):
    os.makedirs(args.s)

epoch = itr
st = time.time()
while epoch < args.n:

    net.train()
    i = 0
    while i < len(datasets):
        dataset = datasets[i]
        i = i+1
        (running_batch, train_batch, isdone, num_running_batch) = get_training_batch(num_running_batch, running_batch, dataset, args, bs)


        if isdone:
            # load
            x1 = train_batch['previmg'].to(device)
            x2 = train_batch['currimg'].to(device)
            y = train_batch['currbb'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = net(x1, x2)
            loss = loss_fn(output, y)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            curr_loss = loss.item()
            end = time.time()
            epoch += 1
            print(f' step : {epoch}/{args.n} | loss : {curr_loss:.4f} | time : {(end-st):.4f} sec per batch ')
            sys.stdout.flush()

            del(train_batch)

            st = time.time()


total_time = time.time() - since
print(f' --- TRAINING IS DONE  IN {total_time // 60:.0f} minutes --- ')


