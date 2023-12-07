# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 10:00:00 2023

"""
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
import MAG2305_torch_public as MAG2305

import torch
from Unetc16 import Unetc16
from Unetc8 import Unetc8
from utils import *
from matplotlib.colors import Normalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MH Test')
    parser.add_argument('--gpu',         type=int,    default=0,         help='GPU ID (default: 0)')

    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',      type=int,    default=1,         help='MAG model layers (default: 1)')

    parser.add_argument('--Ms',          type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',          type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',          type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',        type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',     type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',    type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',    type=Culist, default=(1,1,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',       type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',   type=float,  default=1.0e-5,    help='min error (default: 1.0e-6)')
    parser.add_argument('--max_iter',    type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--sav_samples', type=int,    default=1000,      help='save samples (default: 500)')
    parser.add_argument('--mask',        type=MaskTp, default=False,     help='mask (default: False)')
    parser.add_argument('--Switch_model',type=bool,   default=False,     help='Switch model c16->c8 (default: False)')
    args = parser.parse_args() 
    
    #set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:{}".format(args.gpu))

    ########### Load unet parameter #############
    # load Model
    model_ptc16 = './ckpt/best_model95.6.pt'
    model_ptc8 = './ckpt/best_model483.3.pt'

    model1 = Unetc16().eval().to(device)
    model2 = Unetc8().eval().to(device)

    model1.load_state_dict(torch.load(model_ptc16))
    model2.load_state_dict(torch.load(model_ptc8))

    # initial model
    model = model1
    MAG2305.load_model(model)

    # Switch model from unet c16 to c8
    Switch = args.Switch_model

    #########################
    # Prepare MAG2305 model #
    #########################
    # Create a test-model
    film2 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu)
                            )

    print('Creating {} layer models \n'.format(args.layers))


    #Spin initialization
    spin_split = np.random.randint(low=2, high=32)
    rand_seed = np.random.randint(low=1000, high=100000)
    spin = spin_prepare(spin_split, film2, rand_seed, mask=args.mask)
    film2.SpinInit(spin)

    cell_count = (np.linalg.norm(spin, axis=-1) > 0).sum()

    #####################
    # Update spin state #
    #####################
    print('\nBegin spin updating:\n')
    Hext_range = np.linspace(1000,-1000,201)
    Hext_vec = np.array([np.cos(0.01), np.sin(0.01), 0.0])

    x_plot  = []
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


        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('size:{}_split{}_seed{}\n \nloop:{} , Hext={}'.format(
                    args.w, spin_split, rand_seed, nloop, Hext), fontsize=15)
        fig.subplots_adjust(left=0.05, right=0.9)
        fig.subplots_adjust(wspace=0.5)
        fig.subplots_adjust(hspace=0.3)

        ax1 = fig.add_subplot(2, 4, 1, aspect='equal')
        ax2 = fig.add_subplot(2, 4, 7)
        ax3 = fig.add_subplot(2, 4, 4)
        ax4 = fig.add_subplot(2, 4, 5, aspect='equal')
        ax5 = fig.add_subplot(2, 4, 6, aspect='equal')
        ax6 = fig.add_subplot(2, 4, 8)
        ax7 = fig.add_subplot(2, 4, 2)
        ax8 = fig.add_subplot(2, 4, 3)

        x = np.arange(film2.size[0]) #* film2.cell[0]
        y = np.arange(film2.size[1]) #* film2.cell[1]
        X, Y = np.meshgrid(x, y)


        # Do iteration
        spin_sum2  = np.zeros(3)
        error2_rcd = np.array([])
        Hx2= np.array([])
        Hy2= np.array([])
        mse_list=[]
        mse_Hd_list=[]
        x_mse=[]
        winding_abs_list=[[],[]]
        itern=0
        error_un=1
        lim = 0
        while itern < args.max_iter and error_un > args.error_min:
            
            #Unet
            torch.cuda.synchronize()
            st2  = time.time()
            error_un = film2.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=args.damping)
            spin_un = film2.Spin.detach().cpu()
            torch.cuda.synchronize()
            et2 = time.time()
            
            error2_rcd = np.append(error2_rcd, error_un)
            
            if itern<=100:
                n_print = 1
            elif 100<itern<=1000:
                n_print = 10
            elif 1000<itern:
                n_print = 100

            if itern % n_print ==0:

                print('  Steps: {:<7d}   time cost  un-{:.1e} s '
                        .format(itern+1, et2-st2) )
                print('  un- M_avg={},  error={:.1e}'
                        .format(spin_sum2, error_un) )
                print('')

                spin_wd, winding_abs, winding_sum = winding_density(spin_un[:,:,0,:].view(args.w,args.w,1,3).permute(2, 3, 0, 1))

                winding_abs_list = np.append(winding_abs_list, 
                                            [[itern], [winding_abs]], axis=1)

                #Switching models
                if Switch==True and winding_abs <= 1:
                    model = model2
                MAG2305.load_model(model)

                # Calculate spin sum for MH
                spin_sum2 = spin_un.sum(axis=(0,1,2))
                Hx2 = np.append(Hx2, spin_sum2[0].item())
                Hy2 = np.append(Hy2, spin_sum2[1].item())
                
                # Plot m RGB figures
                ax1.cla()
                spin_plt = spin_un.numpy()
                spin_plt = (spin_plt + 1)/2
                ax1.imshow(spin_plt[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
                ax1.set_title('Spin steps: [{:d}]'.format(itern), fontsize=12)
                ax1.set_xlabel('x [nm]')
                ax1.set_ylabel('y [nm]')
                

                # x-y vector figures
                ax2.cla()
                # 绘制xy轴上的虚线（过零点）
                ax2.axhline(0, linestyle='--', color='gray', linewidth=0.5)
                ax2.axvline(0, linestyle='--', color='gray', linewidth=0.5)
                # 计算矢量箭头的最大长度
                l = torch.max(torch.abs(spin_sum2)).item() * 1.5
                if lim <= l:
                    lim = l
                # 设置x和y轴的范围，保持中心原点
                ax2.set_xlim([-lim, lim])
                ax2.set_ylim([-lim, lim])
                # 保持x和y轴的单位长度相同，保持图的比例
                ax2.set_aspect('equal')
                # 绘制矢量箭头
                ax2.quiver(0, 0, spin_sum2[0], spin_sum2[1], angles='xy', scale_units='xy', scale=1, color='orange', alpha=1.0)
                # 绘制轨迹线
                ax2.plot(Hx2, Hy2, color='orange', alpha=0.5, label='spin_un trails')
                # 设置标题和轴标签
                ax2.set_title('X-Y Vector of trails\nspin_sum:[{:.1e}, {:.1e}, {:.1e}]'
                            .format(spin_sum2.numpy()[0], spin_sum2.numpy()[1], spin_sum2.numpy()[2]), 
                            fontsize=10)
                # 设置轴标签，并避免文字重叠
                ax2.set_xlabel('X-axis', labelpad=10)
                ax2.set_ylabel('Y-axis', labelpad=10)
                # 添加外框
                ax2.spines['left'].set_linewidth(0.5)
                ax2.spines['bottom'].set_linewidth(0.5)
                ax2.spines['right'].set_linewidth(0.5)
                ax2.spines['top'].set_linewidth(0.5)

                
                #MH loop figures
                ax3.cla()
                ax3.plot(x_plot, y2_plot, lw=1.5, label='un', marker='o', markersize=0, color='blue', alpha=1.0)
                ax3.legend(fontsize=12)
                ax3.set_title('M-H data')
                ax3.set_xlabel('Hext [Oe]')
                ax3.set_ylabel('Mext/Ms')
                ax3.set_xlim(min(Hext_range)*1.1, max(Hext_range)*1.1)
                ax3.set_ylim(-1.1, 1.1)

                #Hd rgb figures
                ax4.cla()
                Hd_nn = film2.Hd[:,:,0,:].detach().cpu()
                Hd_un_LL = (Hd_nn).numpy()
                Hd_un_norm = Normalize(vmin=Hd_un_LL.min(), vmax=Hd_un_LL.max())(Hd_un_LL)
                ax4.imshow(Hd_un_norm.transpose(1,0,2), alpha=1.0, origin='lower')
                ax4.set_title('Hd_RGB steps: [{}]'.format(itern), fontsize=12)
                ax4.set_xlabel('x [nm]')
                ax4.set_ylabel('y [nm]')

                #Hd vector graph
                ax5.cla()
                Hd_nn=Hd_nn.numpy()
                ax5.quiver(np.arange(Hd_nn.shape[0]), np.arange(Hd_nn.shape[1]), 
                        Hd_nn[:,:,0].T, Hd_nn[:,:,1].T, Hd_nn[:,:,2].T, 
                        clim=[-0.5, 0.5])
                ax5.set_title('Hd vector Graph', fontsize=12)
                ax5.set_xlabel('x [nm]')
                ax5.set_ylabel('y [nm]')


                # Plot error
                x = np.arange(len(error2_rcd))
                ax6.cla()
                ax6.plot(x, error2_rcd, color='red', alpha=0.6, label='un')
                ax6.set_xlabel('Iterations', fontsize=12)
                ax6.set_ylabel('Maximal $\\Delta$m', fontsize=12)
                ax6.set_yscale('log')
                ax6.legend(fontsize=12)


                #winding_density figures
                ax7.cla()
                spin_wd = spin_wd.detach().cpu().numpy()
                ax7.imshow(spin_wd.T, alpha=1.0, vmin=-0.25, vmax=0.25,  origin='lower')
                ax7.set_title('Wddst Number: abs[{:d}] / [{:d}]'.format(int(winding_abs),int(winding_sum)), fontsize=12)
                ax7.set_xlabel('x [nm]')
                ax7.set_ylabel('y [nm]')

                #winding number figures
                ax8.cla()
                ax8.plot( winding_abs_list[0], winding_abs_list[1], color='green',  alpha=0.7, label='unet' )
                ax8.grid(True, axis='y',lw=1.0, ls='-.')
                ax8.set_xlabel("Iterations")
                ax8.set_ylabel("Absolute winding number")
                #ax8.set_yscale('log')
                ax8.legend()
                ax8.set_title("Vortex number plot\nModel: {}".format(type(model).__name__), fontsize=12)
                
                plt.pause(0.001)
            
            itern += 1
            
        x_plot.append(Hext_val)
        y2_plot.append( np.dot(spin_sum2, Hext_vec)/ cell_count )

        # filename='./figs{}_split{}_seed{}/'.format(args.w, spin_split, rand_seed)
        filename='./figs{}_Ms{}_Ax{}_Ku{}_split{}_seed{}/'.format(args.w, args.Ms, args.Ax, args.Ku, spin_split, rand_seed)
        if not os.path.exists(filename):
            os.makedirs(filename)
        plt.savefig(filename+'loop_{}.png'.format(nloop))
        plt.close()

        # Save MH data
        np.save(filename + "Hext_array", x_plot)
        np.save(filename + "Mext_array_un", y2_plot)
