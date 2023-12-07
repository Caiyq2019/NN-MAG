# -*- coding: utf-8 -*-
import random
import time, os, sys
import numpy as np
from tqdm import tqdm
import argparse

import torch
from utils import *
import MAG2305_torch_public as MAG2305

#nohup python run_data_wHd_Hext_mask_2layers.py --w 32  --Hext_val 100 --mask True --gpu 0 > ./logs/data32_H100_mask.log &
#nohup python run_data_wHd_Hext_mask_2layers.py --w 64  --Hext_val 100 --mask True --gpu 1 > ./logs/data64_H100_mask.log &
#nohup python run_data_wHd_Hext_mask_2layers.py --w 96  --Hext_val 100 --mask True --gpu 0 > ./logs/data96_H100_mask.log &
#nohup python run_data_wHd_Hext_mask_2layers.py --w 128 --Hext_val 100 --mask True --gpu 1 > ./logs/data128_H100_mask.log &


# Custom function to parse a list of floats
def Culist(string):
    try:
        float_values = [float(value) for value in string.split(',')]
        if len(float_values) != 3:
            raise argparse.ArgumentTypeError("List must contain exactly three elements")
        return float_values
    except ValueError:
        raise argparse.ArgumentTypeError("List must contain valid floats")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 1)')

    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,1,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-6,    help='min error (default: 1.0e-6)')
    parser.add_argument('--max_iter',   type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--sav_samples',type=int,    default=1000,      help='save samples (default: 500)')
    parser.add_argument('--mask',       type=bool,   default=False,     help='mask (default: False)')
    args = parser.parse_args() 
    
    #set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:{}".format(args.gpu))

    #################
    # Prepare model #
    #################
    # Create a test-model
    film0 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )
    print('Creating {} layer models \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    film0.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for initializing demag matrix \n'.format(time_finish-time_start))

    ##################
    # External field
    Hext_val = np.random.randn(3)*args.Hext_val
    Hext =  Hext_val* args.Hext_vec


    for seed in tqdm(range(0, 30)):
        path_format = './data_Hd{}_Hext{}_mask/seed{}' if args.mask else './data_Hd{}_Hext{}/seed{}'
        save_path = path_format.format(args.w, int(args.Hext_val), seed)
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        #########################
        # initialize spin state #
        #########################
        spin = initial_spin_prepare(args.w, args.layers, seed)

        #create Mask
        if args.mask==True:
            shape = (args.w, args.w)
            inverse = random.choice([True, False])
            num_points = np.random.randint(low=2, high=shape[0])
            mask = create_random_mask(shape, num_points, inverse)
            spin = film0.SpinInit(spin*mask)  
        else:
            spin = film0.SpinInit(spin)

        #####################
        # Update spin state #
        #####################
        # Do iteration
        Spins_list = []
        Hds_list = []
        error_list = []

        itern = 0
        error_ini = 1

        Spininit = np.reshape(spin[:,:,:,:], (args.w, args.w, args.layers*3))
        Spins_list.append(Spininit)

        while error_ini > args.error_min and itern < args.max_iter:
            error = film0.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=args.damping)  # LLG dynamic method

            error_ini = error
            error_list.append(error)
            itern += 1

            Spins_list.append(np.reshape(film0.Spin.cpu(), (args.w, args.w, args.layers*3)))
            Hds_list.append(np.reshape(film0.Hd.cpu(), (args.w, args.w, args.layers*3)))

        Spins_list.pop(-1)

        #random select n samples
        index = list(range(1, len(Hds_list)-1))
        random.shuffle(index)
        index = sorted(index[:args.sav_samples]) 
        Spins_random_list = [Spins_list[i] for i in index]
        Hds_random_list = [Hds_list[i] for i in index]

        Spins_random_list.append(Spins_list[-1])
        Hds_random_list.append(Hds_list[-1])

        Spins = np.stack(Spins_random_list, axis=0)
        Hds = np.stack(Hds_random_list, axis=0)

        #save data
        np.save(save_path+'/Spins.npy', Spins)
        np.save(save_path+'/Hds.npy', Hds)

        #plot error
        Hext_string=str('[{:.2f}, {:.2f}, {:.2f}]'.format(Hext[0],Hext[1],Hext[2]))
        error_plot(error_list, save_path+'/iterns{:.1e}_errors_{:.1e}'
                   .format(len(error_list), error_list[-1]), Hext_string)

        
