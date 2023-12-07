# -*- coding: utf-8 -*-
"""
Created on Oct Sat 14 18:07:00 2023

#################################
# Check the accuracy of         #
# generated symmetric data:     #
#   imageX,Y, rotate90,180,270  #
#################################
"""
import numpy as np
import MAG2305_torch_public as MAG2305


folder = "./data_Hd64_Hext2_mask/seed100/"
iter_num = 100

#################
# Prepare model #
#################
# Test-model size
w = 64
h = w
test_size = (w,h,2)
model = np.ones(test_size)
cell_count = (model>0).sum()

# Create a test-model
film0 = MAG2305.mmModel(types='bulk', size=test_size, cell=(3,3,3), Ms=1000, Ax=0.5e-6)
# Initialize demag matrix
film0.DemagInit()

##############
# Check data #
##############
# Original
Hd0 = np.load(folder + "Hds.npy")[iter_num].reshape(w,h,2,3)
spin0 = np.load(folder + "Spins.npy")[iter_num].reshape(w,h,2,3)
film0.SpinInit(spin0)
film0.Demag()
Hd = film0.Hd.cpu().numpy()
print("Original check:")
print("  Hd_error = {}".format(abs(Hd - Hd0).sum()/cell_count))
print("")

# X image
Hd0 = np.load(folder + "Hds_imageX.npy")[iter_num].reshape(w,h,2,3)
spin0 = np.load(folder + "Spins_imageX.npy")[iter_num].reshape(w,h,2,3)
film0.SpinInit(spin0)
film0.Demag()
Hd = film0.Hd.cpu().numpy()
print("Image X check:")
print("  Hd_error = {}".format(abs(Hd - Hd0).sum()/cell_count))
print("")
#
# Y image
Hd0 = np.load(folder + "Hds_imageY.npy")[iter_num].reshape(w,h,2,3)
spin0 = np.load(folder + "Spins_imageY.npy")[iter_num].reshape(w,h,2,3)
film0.SpinInit(spin0)
film0.Demag()
Hd = film0.Hd.cpu().numpy()
print("Image Y check:")
print("  Hd_error = {}".format(abs(Hd - Hd0).sum()/cell_count))
print("")
#
# Rotate 90
Hd0 = np.load(folder + "Hds_rotate90.npy")[iter_num].reshape(w,h,2,3)
spin0 = np.load(folder + "Spins_rotate90.npy")[iter_num].reshape(w,h,2,3)
film0.SpinInit(spin0)
film0.Demag()
Hd = film0.Hd.cpu().numpy()
print("Rotate 90 check:")
print("  Hd_error = {}".format(abs(Hd - Hd0).sum()/cell_count))
print("")
#
# Rotate 180
Hd0 = np.load(folder + "Hds_rotate180.npy")[iter_num].reshape(w,h,2,3)
spin0 = np.load(folder + "Spins_rotate180.npy")[iter_num].reshape(w,h,2,3)
film0.SpinInit(spin0)
film0.Demag()
Hd = film0.Hd.cpu().numpy()
print("Rotate 180 check:")
print("  Hd_error = {}".format(abs(Hd - Hd0).sum()/cell_count))
print("")
#
# Rotate 270
Hd0 = np.load(folder + "Hds_rotate270.npy")[iter_num].reshape(w,h,2,3)
spin0 = np.load(folder + "Spins_rotate270.npy")[iter_num].reshape(w,h,2,3)
film0.SpinInit(spin0)
film0.Demag()
Hd = film0.Hd.cpu().numpy()
print("Rotate 270 check:")
print("  Hd_error = {}".format(abs(Hd - Hd0).sum()/cell_count))
print("")
