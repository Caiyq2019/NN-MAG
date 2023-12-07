# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:00:00 2023

#########################################
#                                       #
#  Check Hd calculation from Unet       #
#                                       #
#  -- Compare LLG process controlled    #
#     by MAG2305-Hd or Unet-Hd,         #
#     respectively                      #
#                                       #
#########################################

"""

import torch
import numpy as np
import time, os, sys
import MAG2305_torch_public as MAG2305

from tqdm import tqdm
from plot_utils import *
from vortex_utils import *
from model_utils import *
from nnt.Unet import UNet
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')
    
    parser.add_argument('--krn',        type=int,    default=16,        help='unet first layer kernels (default: 16)')
    #parser.add_argument('--inch',       type=int,   default=3,         help='unet input channels (default: 3)')
    #parser.add_argument('--outch',      type=int,   default=3,         help='unet output channels (default: 3)')
    #parser.add_argument('--trt',        type=bool,  default=False,     help='unet tensorrt (default: False)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 1)')
    parser.add_argument('--split',      type=int,    default=2,         help='MAG model split (default: 1)')
    parser.add_argument('--pre_core',   type=int,    default=10000,     help='MAG model pre_core (default: 0)')
    parser.add_argument('--modelshape', type=str,    default='square',  help='MAG model shape: square, circle, triangle')
    
    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,0,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',      type=float,  default=5.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-5,    help='min error (default: 1.0e-5)')
    parser.add_argument('--max_iter',   type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--nsave',      type=int,    default=10,        help='save number (default: 10)')
    parser.add_argument('--nplot',      type=int,    default=2000,      help='plot number (default: 1000)')
    parser.add_argument('--nsamples',   type=int,    default=100,       help='sample number (default: 1)')
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu))



    #########################
    # Prepare Unet model #
    #########################
    inch=args.layers*3
    model = UNet(c=args.krn, inc=inch, ouc=inch).eval().to(device)
    model.load_state_dict(torch.load('./nnt/k{}/'.format(args.krn) + 'model.pt', map_location=device))

    # creat trt model
    #if args.trt==True:
    #    model = create_trt_model(model, inch, args.w, torch.float16, device)

    # initialize model
    MAG2305.load_model(model)

    #########################
    # Prepare MAG2305 model #
    #########################
    # Model shape
    test_model = create_model(args.w, args.modelshape)

    path0 = "./k{}/size{}/".format(args.krn, args.w)+"pre_core{}/".format(args.pre_core)
    if not os.path.exists(path0):
        os.makedirs(path0)
    np.save(path0 + 'model', test_model[:,:,0])


    # Create a test-model
    film0 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )
    film1 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )

    print('Creating {} layer models \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    film0.DemagInit()
    film1.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))

    sequence = list(range(10000, 110000, 100))[:args.nsamples]
    for rand_seed in tqdm(sequence):
        # Initialize spin state
        time_start = time.time()
        spin0 = MAG2305.get_randspin_2D(size=(args.w, args.w, args.layers),
                                        split=args.split, rand_seed=rand_seed)
        film0.SpinInit(spin0)
        film1.SpinInit(spin0)
        time_finish = time.time()
        print('Time cost: {:f} s for initializing spin state \n'.format(time_finish-time_start))
        
        path = path0+"split{}_rand{}/".format(args.split, rand_seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
        #check any bad cases
        if ( os.path.exists(os.path.join(path, 'Spin_2305_converge.npy'))
            and os.path.exists(os.path.join(path, 'Spin_unet_converge.npy'))
        ):
            continue  # 跳过当前循环或执行其他后续操作
        print('do ',path)

        ###########################
        # Spin update calculation #
        ###########################
        
        print('Begin spin updating:\n')
                
        # External field
        Hext = args.Hext_val * np.array(args.Hext_vec)
    

        # Do iteration
        rcd_dspin_2305 = np.array([[],[]])
        rcd_dspin_unet = np.array([[],[]])
        rcd_dspin_unet_ = np.array([[],[]])
        rcd_windabs_2305 = np.array([[],[]])
        rcd_windabs_unet = np.array([[],[]])
        rcd_windsum_2305 = np.array([[],[]])
        rcd_windsum_unet = np.array([[],[]])
        fig, ax1, ax2, ax3, ax4, ax5, ax6 = plot_prepare()
        
        wind_abs = 10000
        first_iteration = True
        nplot = args.nplot
        time_start = time.time()
        for iters in range(args.max_iter):
            if iters == 0:
                spin_ini = np.array(film0.Spin[:,:,0].cpu())

            error_2305 = film0.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)
            spin_2305 = film0.Spin.cpu().numpy()

            #MAG calculate few steps before unet
            if wind_abs > args.pre_core:
                error_unet = film1.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)
                spin_unet = film1.Spin.cpu().numpy()
                _, wind_abs, _ = get_winding(spin_unet[:,:,0], test_model[:,:,0])
            else:
                error_unet = film1.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=0.1)
                spin_unet = film1.Spin.cpu().numpy()

                #plot figure
                if first_iteration:
                    nplot = iters
                    first_iteration = False  # Set the flag to False after the first iteration
                else:
                    nplot = args.nplot

                
            if iters % args.nsave ==0 or max(error_2305,error_unet)<=args.error_min:
                rcd_dspin_2305 = np.append(rcd_dspin_2305, [[iters], [error_2305]], axis=1)
                rcd_dspin_unet = np.append(rcd_dspin_unet, [[iters], [error_unet]], axis=1)

                if wind_abs <= args.pre_core:
                    rcd_dspin_unet_ = np.append(rcd_dspin_unet_, [[iters], [error_unet]], axis=1)
        
                wind_dens_2305, wind_abs_2305, wind_sum_2305 = get_winding(spin_2305[:,:,0],
                                                                          test_model[:,:,0])
                wind_dens_unet, wind_abs_unet, wind_sum_unet = get_winding(spin_unet[:,:,0],
                                                                          test_model[:,:,0])
                rcd_windabs_2305 = np.append(rcd_windabs_2305, 
                                            [[iters], [wind_abs_2305]], axis=1)
                rcd_windabs_unet = np.append(rcd_windabs_unet, 
                                            [[iters], [wind_abs_unet]], axis=1)
                rcd_windsum_2305 = np.append(rcd_windsum_2305, 
                                            [[iters], [wind_sum_2305]], axis=1)
                rcd_windsum_unet = np.append(rcd_windsum_unet, 
                                            [[iters], [wind_sum_unet]], axis=1)
        
            if iters % nplot ==0 or max(error_2305,error_unet)<=args.error_min:
                plot_spin( spin_2305[:,:,0], ax1, '2305 - iters{}'.format(iters))
                plot_spin( spin_unet[:,:,0], ax4, 'Unet - iters{}'.format(iters))
                plot_wind( wind_dens_2305, ax2, '2305-vortices wd[{}]/[{}]'.format(round(wind_abs_2305), round(wind_sum_2305)))
                plot_wind( wind_dens_unet, ax5, 'Unet-vortices wd[{}]/[{}]'.format(round(wind_abs_unet), round(wind_sum_unet)))
                compare_error( rcd_dspin_2305, rcd_dspin_unet, ax3 )
                compare_wind( rcd_windabs_2305, rcd_windabs_unet, ax6 )
        
                # plot_show(0.1)
                plot_save(path, "spin_iters{}".format(iters))
        
            if max(error_2305,error_unet)<=args.error_min or iters==args.max_iter-1:
                spin_end_2305 = np.array(film0.Spin[:,:,0].cpu())
                spin_end_unet = np.array(film1.Spin[:,:,0].cpu())
                plot_close()
                break
        
        
        time_finish = time.time()
        print('End one case. Steps - {} ; Time cost - {:.1f}s\n'
                .format(iters, time_finish-time_start) )
        
        ###################
        # Data processing #
        ###################
        
        np.save(path+'Dspin_2305_max', rcd_dspin_2305)
        np.save(path+'Dspin_unet_max', rcd_dspin_unet_)
        np.save(path+'Wind_2305_abs', rcd_windabs_2305)
        np.save(path+'Wind_unet_abs', rcd_windabs_unet)
        np.save(path+'Wind_2305_sum', rcd_windsum_2305)
        np.save(path+'Wind_unet_sum', rcd_windsum_unet)
        
        np.save(path+'Spin_initial',  spin_ini)
        np.save(path+'Spin_2305_converge', spin_end_2305)
        np.save(path+'Spin_unet_converge', spin_end_unet)
