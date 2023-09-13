#!/usr/bin/env python
#CARE IF PYTHON OR PYTHON 3
# coding: utf-8


#/******************************************
#*MIT License
#*
# *Copyright (c) [2023] [Giuseppe Sorrentino, Marco Venere, Davide Conficconi, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *Copyright (c) [2022] [Eleonora D'Arnese, Davide Conficconi, Emanuele Del Sozzo, Luigi Fusco, Donatella Sciuto, Marco Domenico Santambrogio]
# *Copyright (c) [2020] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Donatella Sciuto, Marco D. Santambrogio]

#*Permission is hereby granted, free of charge, to any person obtaining a copy
#*of this software and associated documentation files (the "Software"), to deal
#*in the Software without restriction, including without limitation the rights
#*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#*copies of the Software, and to permit persons to whom the Software is
#*furnished to do so, subject to the following conditions:
#*
#*The above copyright notice and this permission notice shall be included in all
#*copies or substantial portions of the Software.
#*
#*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#*SOFTWARE.
#*/
import re
import os
import pydicom
import cv2
import numpy as np
import math
import glob
import time
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import gc
from pynq import Overlay
import pynq
from pynq import allocate
import struct
import statistics
import argparse
#import hephaestusAccelerators
#import hephaestusAcceleratorsMapper as hephaestus
import hephaestusNewFunctions as hephaestusNewFunctions
import hephaestusWonderfulListMapper as hephaestus
from hephaestusImageRegistration import *

def optimize_goldsearch(par, rng, ref_sup_ravel, flt_sup, volume, linear_par,i):
    start=par-0.382*rng
    end=par+0.618*rng
    c=(end-(end-start)/1.618)
    d=(start+(end-start)/1.618)
    best_mi = 0.0
    while(math.fabs(c-d)>0.005):
        linear_par[i]=c
        a=to_matrix_complete(linear_par)
        linear_par[i]=d
        b=to_matrix_complete(linear_par)

        transformed = transform(flt_sup, a, volume)
        transformed = torch.from_numpy(transformed)
        mi_a= mutual_info_sw(volume, ref_sup_ravel, transformed, 512) * -1
        transformed2 = transform(flt_sup, b, volume)
        transformed2 = torch.from_numpy(transformed2)
        mi_b= mutual_info_sw(volume, ref_sup_ravel, transformed2, 512) * -1

        if(mi_a < mi_b):
            end=d
            best_mi = mi_a
            linear_par[i]=c
        else:
            start=c
            best_mi = mi_b
            linear_par[i]=d

        c=(end-(end-start)/1.618)
        d=(start+(end-start)/1.618)

    return (end+start)/2, best_mi

def mutual_info_sw(volume, Ref_uint8, Flt_uint8, dim):
        j_h=np.histogram2d(torch.ravel(Ref_uint8).numpy(),torch.ravel(Flt_uint8).numpy(),bins=[256,256])[0]
        j_h=j_h/(volume*dim*dim)
          
        j_h1=j_h[np.where(j_h>0.000000000000001)]
        entropy=(np.sum(j_h1*np.log2(j_h1)))*-1

        href=np.sum(j_h,axis=0)
        hflt=np.sum(j_h,axis=1)     

        href=href[np.where(href>0.000000000000001)]
        eref=(np.sum(href*(np.log2(href))))*-1

        hflt=hflt[np.where(hflt>0.000000000000001)]
        eflt=(sum(hflt*(np.log2(hflt))))*-1

        mutualinfo=eref+eflt-entropy

        return(mutualinfo)

def optimize_powell(rng, par_lin, ref_sup_ravel, flt_sup, volume):
    converged = False
    eps = 0.000005
    last_mut=100000.0
    it=0
    while(not converged):
        converged=True
        it=it+1
        for i in range(len(par_lin)):
            cur_par = par_lin[i]
            cur_rng = rng[i]
            param_opt, cur_mi = optimize_goldsearch(cur_par, cur_rng, ref_sup_ravel, flt_sup, volume, par_lin,i)
            par_lin[i]=cur_par
            if last_mut-cur_mi>eps:
                par_lin[i]=param_opt
                last_mut=cur_mi
                converged=False
            else:
                par_lin[i]=cur_par
    return (par_lin)



def register_images(Ref_uint8, Flt_uint8, volume, filename):
    start_single_sw = time.time()
    params = np.zeros((3,4))
    params = estimate_initial(Ref_uint8, Flt_uint8, params, volume)
    rng=np.array([80.0, 80.0, 20.0, 1.0, 1.0, 1.0])
    pa=[params[0][3], params[1][3], 0.0, params[0][0], 1.0, 1.0]
    
    Ref_uint8_ravel = torch.ravel(Ref_uint8)
    #print(Ref_uint8_ravel.shape)
    optimal_params = optimize_powell(rng, pa, Ref_uint8_ravel, Flt_uint8,  volume) 
    params_trans=to_matrix_complete(optimal_params)
    flt_transform = transform(Flt_uint8, params_trans, volume)
    end_single_sw = time.time()
    print('Final time: ', end_single_sw - start_single_sw)
    with open(filename, 'a') as file2:
                file2.write("%s\n" % (end_single_sw - start_single_sw))
 
    return (flt_transform)

def compute_wrapper(args, hephaestus_pl, num_threads=1):
    config=args.config
    the_list_of_wonderful_lists=hephaestus.hephaestus_accel_map(programmable_logic=hephaestus_pl, \
        image_dimension=args.image_dimension, volume = args.volume, platform=args.platform)
    for k in range(args.offset, args.patient):
        pool = []
        curr_prefix = args.prefix+str(k)
        curr_ct = os.path.join(curr_prefix,args.ct_path)
        curr_pet = os.path.join(curr_prefix,args.pet_path)
        curr_res = os.path.join("",args.res_path)
        os.makedirs(curr_res,exist_ok=True)
        CT=glob.glob(curr_ct+'/*dcm')
        PET=glob.glob(curr_pet+'/*dcm')
        PET.sort(key=lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
        CT.sort(key=lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
        assert len(CT) == len(PET)
        wonderful_list = the_list_of_wonderful_lists[0]
        compute(CT, PET, args.volume, args.filename, curr_res, 0, k, wonderful_list, args.first_slice, args.last_slice, args.num_subvolumes)
        


def compute(CT, PET, volume, filename, curr_res, patient_id, wonderful_list, first_slice, last_slice, num_subvolumes):
    final_img=[]
    times=[]
    t = 0.0
    it_time = 0.0
    hist_dim = 256
    dim = 512
    refs = []
    flts = []
    couples = 0
    left = first_slice
    right = last_slice
    N = num_subvolumes
    for c,ij in enumerate(zip(CT[left:right], PET[left:right])):
        i = ij[0]
        j = ij[1]

        ref = pydicom.dcmread(i)
        Ref_img = torch.tensor(ref.pixel_array.astype(np.int16), dtype=torch.int16, device="cpu")
        Ref_img[Ref_img==-2000]=1

        flt = pydicom.dcmread(j)
        Flt_img = torch.tensor(flt.pixel_array.astype(np.int16), dtype=torch.int16, device="cpu")

        Ref_img = (Ref_img - Ref_img.min())/(Ref_img.max() - Ref_img.min())*255
        Ref_uint8 = Ref_img.round().type(torch.uint8)

        Flt_img = (Flt_img - Flt_img.min())/(Flt_img.max() - Flt_img.min())*255
        Flt_uint8 = Flt_img.round().type(torch.uint8)

        refs.append(Ref_uint8)
        flts.append(Flt_uint8)
        del ref
        del flt
        del Flt_img
        del Ref_img
        gc.collect()
        couples = couples + 1
        if couples >= len(CT[left:right]):
            break

    refs3D = torch.cat(refs)
    flts3D = torch.cat(flts)
    refs3D = torch.reshape(refs3D,(len(CT[left:right]),512,512))
    flts3D = torch.reshape(flts3D,(len(CT[left:right]),512,512))
    del refs
    del flts
    gc.collect()

    transform_matrix = register_images(refs3D, flts3D, len(CT[left:right]), filename)
    #print('Trasformata:')
    #print(transform_matrix)

    for index in range(N):
        couples = 0
        refs = []
        flts = []
        for c,ij in enumerate(zip(CT[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))], PET[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))])):
            i = ij[0]
            j = ij[1]

            ref = pydicom.dcmread(i)
            Ref_img = torch.tensor(ref.pixel_array.astype(np.int16), dtype=torch.int16, device="cpu")
            Ref_img[Ref_img==-2000]=1

            flt = pydicom.dcmread(j)
            Flt_img = torch.tensor(flt.pixel_array.astype(np.int16), dtype=torch.int16, device="cpu")

            Ref_img = (Ref_img - Ref_img.min())/(Ref_img.max() - Ref_img.min())*255
            Ref_uint8 = Ref_img.round().type(torch.uint8)

            Flt_img = (Flt_img - Flt_img.min())/(Flt_img.max() - Flt_img.min())*255
            Flt_uint8 = Flt_img.round().type(torch.uint8)
            refs.append(Ref_uint8)
            flts.append(Flt_uint8)
            del ref
            del flt
            del Flt_img
            del Ref_img
            gc.collect()
            couples = couples + 1
            #if couples >= int(volume / N):
            #    break
        refs3D = torch.cat(refs)
        flts3D = torch.cat(flts)
        refs3D = torch.reshape(refs3D,(len(CT[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))]),512,512))
        flts3D = torch.reshape(flts3D,(len(CT[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))]),512,512))
        del refs
        del flts
        gc.collect()
        flt_transform = transform(flts3D, transform_matrix, len(PET[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))]))
        save_data(flt_transform, PET[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))], curr_res)
     

        








def main():

    parser = argparse.ArgumentParser(description='Hephaestus software for IR onto a python env')
    parser.add_argument("-pt", "--patient", nargs='?', help='Number of the patient to analyze', default=1, type=int)
    parser.add_argument("-o", "--offset", nargs='?', help='Starting patient to analyze', default=0, type=int)
    parser.add_argument("-cp", "--ct_path", nargs='?', help='Path of the CT Images', default='./')
    parser.add_argument("-pp", "--pet_path", nargs='?', help='Path of the PET Images', default='./')
    parser.add_argument("-rp", "--res_path", nargs='?', help='Path of the Results', default='./')
    parser.add_argument("-ol", "--overlay", nargs='?', help='Path and filename of the target overlay', default='./hephaestus_wrapper.bit')
    parser.add_argument("-clk", "--clock", nargs='?', help='Target clock frequency of the PL', default=100, type=int)
    parser.add_argument("-px", "--prefix", nargs='?', help='prefix Path of patients folder', default='./')
    parser.add_argument("-p", "--platform", nargs='?', help='platform to target.\
     \'Alveo\' is used for PCIe/XRT based,\n while \'Ultra96\' will setup for a Zynq-based environment', default='Alveo')
    parser.add_argument("-mem", "--caching", action='store_true', help='if it use or not the caching')    
    parser.add_argument("-im", "--image_dimension", nargs='?', help='Target images dimensions', default=512, type=int)
    parser.add_argument("-c", "--config", nargs='?', help='prefix Path of patients folder', default='./')
    parser.add_argument("-vol", "--volume", nargs='?', help="Volume of the image to analyze, expressed as number of slices", default=1, type=int)
    parser.add_argument("-fs", "--first_slice", nargs='?', help="Index of the first slice of the subvolume to register, starting from 0.", default=0, type=int)
    parser.add_argument("-ls", "--last_slice", nargs='?', help="Index of the last slice of the subvolume to register, starting from 0.", default=-1, type=int)
    parser.add_argument("-ns", "--num_subvolumes", nargs='?', help="Number of disjoint subvolume for which to apply the final registration.", default=1, type=int)
    parser.add_argument("-f", "--filename", nargs='?', help='Name of the file in which to write times', default='test.csv')

    

    args = parser.parse_args()

    if args.last_slice == -1:
         args.last_slice = args.volume

    if args.first_slice < 0 or args.last_slice < args.first_slice or args.last_slice > args.volume or args.num_subvolumes < 1 or args.num_subvolumes > args.volume:
        raise ValueError("Wrong parameter values!")

  
        
    


    num_threads = 1

    if args.platform=='Zynq' :
        from pynq.ps import Clocks
        print("Previous Frequency "+str(Clocks.fclk0_mhz))
        Clocks.fclk0_mhz = args.clock 
        print("New frequency "+str(Clocks.fclk0_mhz))

    print(args.config)
    print(args)

    compute_wrapper(args, num_threads)


    print("hephaestus Powell python is at the end :)")
        

if __name__== "__main__":
    main()
