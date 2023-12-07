# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 10:00:00 2023

"""
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from matplotlib.colors import Normalize

import torch
from utils import *

import MAG2305_torch_public as MAG2305





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MH Test')
    parser.add_argument('--gpu',         type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--krn',         type=int,    default=16,        help='unet first layer kernels (default: 16)')

    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',      type=int,    default=2,         help='MAG model layers (default: 1)')

    parser.add_argument('--Ms',          type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',          type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',          type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',        type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',     type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',    type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',    type=Culist, default=(1,1,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',       type=float,  default=5.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',   type=float,  default=1.0e-5,    help='min error (default: 1.0e-6)')
    parser.add_argument('--max_iter',    type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--mask',        type=MaskTp, default=False,     help='mask (default: False)')
    args = parser.parse_args() 
    
    #set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:{}".format(args.gpu))

    ########### Load unet parameter #############
    # load Model

    from nnt.Unet import UNet
    model = UNet(c=args.krn, inc=args.layers*3, ouc=args.layers*3).eval().to(device)

    ckpt = './nnt/k{}/model.pt'.format(args.krn)
    #model.load_state_dict(torch.load(ckpt, map_location=device))
    
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # initial model
    MAG2305.load_model(model)


    #########################
    # Prepare MAG2305 model #
    #########################
    # Create a test-model
    film1 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )
    
    film2 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )

    print('Creating {} layer models \n'.format(args.layers))


    #Spin initialization
    spin_split = np.random.randint(low=2, high=32)
    rand_seed  = np.random.randint(low=1000, high=100000)
    spin = spin_prepare(spin_split, film1, rand_seed, mask=args.mask)
    film1.SpinInit(spin)
    film2.SpinInit(spin)

    cell_count = (np.linalg.norm(spin, axis=-1) > 0).sum()

    # Initialize demag matrix
    time_start = time.time()
    film1.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for initializing demag matrix \n'.format(time_finish-time_start))

    #####################
    # Update spin state #
    #####################
    print('\nBegin spin updating:\n')
    Hext_range = np.append( np.linspace(0,1000,10, endpoint=False), 
                        np.linspace(1000,-1000,201) )
    
    #Hext_range = np.linspace(1000,-1000,201)
    Hext_vec = np.array([np.cos(0.01), np.sin(0.01), 0.0])

    x_plot  = []
    y1_plot = []
    y2_plot = []
    total_step = []
    for nloop, Hext_val in enumerate(Hext_range):
        Hext = Hext_val * Hext_vec

        # Create a canvas
        parameters = {'axes.labelsize' : 10,
                      'axes.titlesize' : 10,
                      'xtick.labelsize': 10,
                      'ytick.labelsize': 10
                     }
        plt.rcParams.update(parameters)


        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('{} layers film size:{}_split{}_seed{}\n \nloop:{} , Hext={}'.format(
                    args.layers, args.w, spin_split, rand_seed, nloop, Hext), fontsize=12 )
        fig.subplots_adjust(left=0.05, right=0.9)
        fig.subplots_adjust(wspace=0.5)
        fig.subplots_adjust(hspace=0.5)

        ax1 = fig.add_subplot(3, 4, 1) #spin mm
        ax2 = fig.add_subplot(3, 4, 2) #spin un
        ax3 = fig.add_subplot(3, 4, 3) #mse plot
        ax4 = fig.add_subplot(3, 4, 4) #MH loop

        ax5 = fig.add_subplot(3, 4, 5) #Hd mm
        ax6 = fig.add_subplot(3, 4, 6) #Hd un
        ax7 = fig.add_subplot(3, 4, 7) #mse plot
        ax8 = fig.add_subplot(3, 4, 8) #xy vector plot
        
        ax9 = fig.add_subplot(3, 4, 9)  #winding density mm
        ax10= fig.add_subplot(3, 4, 10) #winding density un
        ax11= fig.add_subplot(3, 4, 11) #winding number plot
        ax12= fig.add_subplot(3, 4, 12) #error plot


        x = np.arange(film2.size[0]) #* film2.cell[0]
        y = np.arange(film2.size[1]) #* film2.cell[1]
        X, Y = np.meshgrid(x, y)

        # Do iteration
        spin_sum1  = np.zeros(3)
        spin_sum2  = np.zeros(3)

        error1_rcd = np.array([])
        error2_rcd = np.array([])

        Hx1= np.array([])
        Hy1= np.array([])
        Hx2= np.array([])
        Hy2= np.array([])

        mse_list=[]
        mse_Hd_list=[]
        x_mse=[]
        winding_abs_list=[[],[],[]]
        itern=0
        error_un=1
        error_mm=1
        error_fluc = 1
        lim = 0
        while itern < args.max_iter and max(error_un, error_mm) > args.error_min:
            
            #MAG
            torch.cuda.synchronize()
            st1  = time.time()
            error_mm = film1.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)
            torch.cuda.synchronize()
            et1 = time.time()

            #Unet
            torch.cuda.synchronize()
            st2  = time.time()
            error_un = film2.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=0.1)
            torch.cuda.synchronize()
            et2 = time.time()
            
            spin_mm = film1.Spin.detach().cpu()
            spin_un = film2.Spin.detach().cpu()

            error1_rcd = np.append(error1_rcd, error_mm)
            error2_rcd = np.append(error2_rcd, error_un)
            
            # Check error break condition
            if itern>5000:
                error_fluc = np.abs(error2_rcd[-2000:].mean()-error2_rcd[-500:].mean()) / error2_rcd[-2000:].mean()
                if error_fluc<0.02 and error_mm <= args.error_min:
                    print('Unet error not decrease! break!')
                    break

            if itern<=100:
                n_print = 1
            elif 100<itern<=1000:
                n_print = 10
            elif 1000<itern:
                n_print = 100

            if itern % n_print ==0:

                print('  Steps: {:<7d}   time cost mm-{:.1e} s '
                        .format(itern+1, et1-st1) )
                print('                             un-{:.1e} s '
                        .format(et2-st2) )
                print('                            ratio: {:.1e} '.format((et1-st1)/(et2-st2+1e-8)))
                
                print('  mm- M_avg={},  error={:.1e}'
                        .format(spin_sum1, error_mm) )
                print('  un- M_avg={},  error={:.1e}'
                        .format(spin_sum2, error_un) )
                print('')

                spin_wd1, winding_abs1, winding_sum1 = winding_density(spin_mm[:,:,0,:].view(args.w,args.w,1,3).permute(2, 3, 0, 1))
                spin_wd2, winding_abs2, winding_sum2 = winding_density(spin_un[:,:,0,:].view(args.w,args.w,1,3).permute(2, 3, 0, 1))

                winding_abs_list = np.append(winding_abs_list, 
                                            [[itern], [winding_abs1], [winding_abs2]], axis=1)


                # Calculate spin sum for MH
                spin_sum1 = spin_mm.sum(axis=(0,1,2))
                Hx1 = np.append(Hx1, spin_sum1[0].item())
                Hy1 = np.append(Hy1, spin_sum1[1].item())

                spin_sum2 = spin_un.sum(axis=(0,1,2))
                Hx2 = np.append(Hx2, spin_sum2[0].item())
                Hy2 = np.append(Hy2, spin_sum2[1].item())
                
                # Plot spin-mm RGB figures
                ax1.cla()
                spin_plt = spin_mm.numpy()
                spin_plt = (spin_plt + 1)/2
                ax1.imshow(spin_plt[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
                ax1.set_title('Spin-mm steps: [{:d}]'.format(itern), fontsize=12)
                ax1.set_xlabel('x [nm]')
                ax1.set_ylabel('y [nm]')

                # Plot spin-mm RGB figures
                ax2.cla()
                spin_plt = spin_un.numpy()
                spin_plt = (spin_plt + 1)/2
                ax2.imshow(spin_plt[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
                ax2.set_title('Spin-un steps: [{:d}]'.format(itern), fontsize=12)
                ax2.set_xlabel('x [nm]')
                ax2.set_ylabel('y [nm]')

                # Plot mse heatmap
                ax3.cla()
                mse = np.square(spin_un.numpy() - spin_mm.numpy()).sum(axis=-1)
                mse = mse.transpose((1, 0, 2))[:,:,0]
                im = ax3.imshow(mse, cmap='hot', origin="lower")
                ax3.set_title('MSE of spin_mm & spin_un, \nMSE_avg: {:4f}'.format(mse.mean()), fontsize=10)
                if itern != 0:
                    cbar1.update_normal(im)
                else:
                    cbar1 = fig.colorbar(im, ax=ax3)


                #MH loop figures
                ax4.cla()
                ax4.plot(x_plot, y1_plot, lw=1.5, label='mm', marker='o', markersize=0, color='blue',  alpha=0.6)
                ax4.plot(x_plot, y2_plot, lw=1.5, label='un', marker='o', markersize=0, color='red', alpha=0.6)
                ax4.legend(fontsize=10, loc='upper left')
                ax4.set_title('M-H data')
                ax4.set_xlabel('Hext [Oe]')
                ax4.set_ylabel('Mext/Ms')
                ax4.set_xlim(min(Hext_range)*1.1, max(Hext_range)*1.1)
                ax4.set_ylim(-1.1, 1.1)
                ax4.grid(True, axis='both', lw=0.5, ls='-.')

                #row 2
                #Hd-mm rgb figures
                ax5.cla()
                Hd_mm = film1.Hd[:,:,0,:].detach().cpu().numpy()
                Hd_mm_norm = Normalize(vmin=Hd_mm.min(), vmax=Hd_mm.max())(Hd_mm)
                ax5.imshow(Hd_mm_norm.transpose(1,0,2), alpha=1.0, origin='lower')
                ax5.set_title('Hd_mm steps: [{}]'.format(itern), fontsize=12)
                ax5.set_xlabel('x [nm]')
                ax5.set_ylabel('y [nm]')

                #Hd-un rgb figures
                ax6.cla()
                Hd_un = film2.Hd[:,:,0,:].detach().cpu().numpy()
                Hd_un_norm = Normalize(vmin=Hd_un.min(), vmax=Hd_un.max())(Hd_un)
                ax6.imshow(Hd_un_norm.transpose(1,0,2), alpha=1.0, origin='lower')
                ax6.set_title('Hd_un steps: [{}]'.format(itern), fontsize=12)
                ax6.set_xlabel('x [nm]')
                ax6.set_ylabel('y [nm]')

                # Plot mse heatmap
                ax7.cla()
                mse_Hd = np.square(Hd_mm - Hd_un).sum(axis=-1)
                im = ax7.imshow(mse_Hd.T, cmap='hot', origin="lower")
                ax7.set_title('MSE of Hd_mm & Hd_un \nMSE_avg{:.1e}'.format(mse_Hd.mean()), fontsize=10)
                if itern != 0:
                    cbar2.update_normal(im)
                else:
                    cbar2 = fig.colorbar(im, ax=ax7)


                # x-y vector figures
                ax8.cla()
                # 绘制xy轴上的虚线（过零点）
                ax8.axhline(0, linestyle='--', color='gray', linewidth=0.5)
                ax8.axvline(0, linestyle='--', color='gray', linewidth=0.5)
                l = 1.1 * max(torch.abs(spin_sum1).max().item(), torch.abs(spin_sum2).max().item())
                if lim <= l:
                    lim = l
                ax8.set_xlim([-lim, lim])
                ax8.set_ylim([-lim, lim])
                ax8.set_aspect('equal')
                # 绘制矢量箭头
                ax8.quiver(0, 0, spin_sum1[0], spin_sum1[1], angles='xy', scale_units='xy', scale=1, color='green',  alpha=1.0)
                ax8.quiver(0, 0, spin_sum2[0], spin_sum2[1], angles='xy', scale_units='xy', scale=1, color='orange', alpha=1.0)
                # 绘制轨迹线
                ax8.plot(Hx1, Hy1, color='green',  alpha=0.5, label='spin_mm trails')
                ax8.plot(Hx2, Hy2, color='orange', alpha=0.5, label='spin_un trails')
                
                text = 'mm:[{:.2f}, {:.2f}, {:.3f}] \n un: [{:.2f}, {:.2f}, {:.3f}]'.format(
                    spin_sum1.numpy()[0], spin_sum1.numpy()[1], spin_sum1.numpy()[2],
                    spin_sum2.numpy()[0], spin_sum2.numpy()[1], spin_sum2.numpy()[2]
                )
                ax8.set_title('X-Y Vector of spinsum trails\n{}'.format(text),fontsize=8)
                ax8.set_xlabel('X-axis', labelpad=10)
                ax8.set_ylabel('Y-axis', labelpad=10)
                ax8.legend(fontsize=8, loc='upper left')

                ax8.spines['left'].set_linewidth(0.5)
                ax8.spines['bottom'].set_linewidth(0.5)
                ax8.spines['right'].set_linewidth(0.5)
                ax8.spines['top'].set_linewidth(0.5)
                
                #row 3
                #winding_density figures
                ax9.cla()
                spin_wd1 = spin_wd1.detach().cpu().numpy()
                ax9.imshow(spin_wd1.T, alpha=1.0, vmin=-0.25, vmax=0.25,  origin='lower')
                ax9.set_title('Wdd-mm Number: abs[{:d}] / [{:d}]'.format(int(winding_abs1),int(winding_sum1)), fontsize=10)
                ax9.set_xlabel('x [nm]')
                ax9.set_ylabel('y [nm]')
                
                #winding_density figures
                ax10.cla()
                spin_wd2 = spin_wd2.detach().cpu().numpy()
                ax10.imshow(spin_wd2.T, alpha=1.0, vmin=-0.25, vmax=0.25,  origin='lower')
                ax10.set_title('Wdd-un Number: abs[{:d}] / [{:d}]'.format(int(winding_abs2),int(winding_sum2)), fontsize=10)
                ax10.set_xlabel('x [nm]')
                ax10.set_ylabel('y [nm]')

                #winding number figures
                ax11.cla()
                ax11.plot( winding_abs_list[0], winding_abs_list[1], color='green',  alpha=0.7, label='mm' )
                ax11.plot( winding_abs_list[0], winding_abs_list[2], color='orange', alpha=0.7, label='un' )
                ax11.grid(True, axis='y',lw=1.0, ls='-.')
                ax11.set_xlabel("Iterations")
                ax11.set_ylabel("Absolute winding number")
                #ax11.set_yscale('log')
                ax11.legend(fontsize=10)
                ax11.set_title("Vortex number plot\nModel: [{} k{}]".format(type(model).__name__,args.krn), fontsize=12)

                # Plot error
                x = np.arange(len(error2_rcd))
                ax12.cla()
                ax12.plot(x, error1_rcd, color='blue', alpha=0.6, label='mm')
                ax12.plot(x, error2_rcd, color='red',  alpha=0.6, label='un')
                ax12.set_title('Error plot, fluctuate:[{:.2f}]'.format(error_fluc), fontsize=10)
                ax12.set_xlabel('Iterations', fontsize=12)
                ax12.set_ylabel('Maximal $\\Delta$m', fontsize=12)
                ax12.set_yscale('log')
                ax12.legend(fontsize=10, loc='upper right')
                
                plt.pause(0.001)
            
            itern += 1

        #MH loop data
        x_plot.append(Hext_val)
        y1_plot.append( np.dot(spin_sum1, Hext_vec)/ cell_count )
        y2_plot.append( np.dot(spin_sum2, Hext_vec)/ cell_count )

        filename='./figs_k{}/shape_{}/size{}_Ms{}_Ax{}_Ku{}_dtime{}_split{}_seed{}_Layers{}/'.format(
                    args.krn, args.mask, args.w, 
                    args.Ms, args.Ax, args.Ku, 
                    args.dtime, spin_split, rand_seed, args.layers
                    )
        
        if not os.path.exists(filename):
            os.makedirs(filename)
        plt.savefig(filename+'loop_{}.png'.format(nloop))
        plt.close()

        # Save MH data
        np.save(filename + "Hext_array", x_plot)
        np.save(filename + "Mext_array_mm", y1_plot)
        np.save(filename + "Mext_array_un", y2_plot)
