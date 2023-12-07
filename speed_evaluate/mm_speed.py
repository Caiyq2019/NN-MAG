# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 10:00:00 2023

"""

import numpy as np
import time, os, random, sys
import argparse

import torch
from utils import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test')
    parser.add_argument('--gpu',        type=int,   default=0,         help='GPU ID (default: 0)')
    
    parser.add_argument('--krn',        type=int,   default=16,        help='unet first layer kernels (default: 16)')
    parser.add_argument('--inch',       type=int,   default=3,         help='unet input channels (default: 3)')
    parser.add_argument('--outch',      type=int,   default=3,         help='unet output channels (default: 3)')
    parser.add_argument('--ckpt',       type=str,   default=None,      help='unet checkpoint (default: ./ckpt/best_model.pt)')
    parser.add_argument('--trt',        type=bool,  default=False,     help='unet tensorrt (default: False)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=1,         help='MAG model layers (default: 1)')

    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,0,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-5,    help='min error (default: 1.0e-5)')
    parser.add_argument('--max_iter',   type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--n_loop',     type=int,    default=100,       help='loop number (default: 100)')
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu))



    #########################
    # Prepare MAG2305 model #
    #########################
    if args.layers == 1:
        import MAG2305_torch_2D as MAG2305
    else:
        import MAG2305_torch_public as MAG2305

    # Create a test-model
    film1 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )
    print('Creating {} layer models \n'.format(args.layers))


    # Initialize demag matrix
    time_start = time.time()
    film1.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))

    # spin initialization cases
    spin_split = np.random.randint(low=2, high=32)
    rand_seed = np.random.randint(low=0, high=100000)
    spin = spin_prepare(spin_split, film1, rand_seed)
    film1.SpinInit(spin)
    print('spin shape',film1.Spin.shape)

    # External field
    Hext = args.Hext_val * np.array(args.Hext_vec)

    #####################
    # total step speed #
    #####################

    #MAG2305
    a1=torch.zeros(args.n_loop).to(device)
    for i in range(args.n_loop):
        torch.cuda.synchronize()
        st1  = time.time()
        error_mm = film1.SpinLLG_RK4( Hext=Hext, dtime=args.dtime, damping=args.damping )
        torch.cuda.synchronize()
        et1 = time.time()
        a1[i]=et1-st1

    Spin_speed = torch.mean(a1[10:]).item()

    #####################
    #     Hd speed      #
    #####################
    #MAG2305
    a2=torch.zeros(args.n_loop).to(device)
    for i in range(args.n_loop):
        torch.cuda.synchronize()
        st1  = time.time()
        _ = film1.Demag()
        torch.cuda.synchronize()
        et1 = time.time()
        a2[i]=et1-st1

    Hd_speed = torch.mean(a2[10:]).item()
    print('||MAG_size: {} || total step speed: {:.1e} s || Hd speed: {:.1e} s||'.format(args.w, Spin_speed, Hd_speed))
