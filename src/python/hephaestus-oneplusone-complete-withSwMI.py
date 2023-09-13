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

import os
import re
import pydicom
import cv2
import numpy as np
import math
import glob
import time
import pandas as pd
import multiprocessing
import struct
import statistics
import argparse
import gc

import hephaestusNewFunctions
import hephaestusWonderfulListMapper as hephaestus
from hephaestusImageRegistration import *

m_Gaussfaze=1
m_Gausssave=np.zeros((1,8*128))
dat=np.zeros((1,6))
m_GScale=1.0/30000000.0


def NormalVariateGenerator():

    global m_Gaussfaze,m_Gausssave,m_GScale
    m_Gaussfaze = m_Gaussfaze-1
    if (m_Gaussfaze):
        return m_GScale * m_Gausssave[m_Gaussfaze]
    else:
        return FastNorm()



def SignedShiftXOR(x):
    uirs = np.uint32(x)
    c=np.int32((uirs << 1) ^ 333556017) if np.int32(x <= 0) else np.int32(uirs << 1)
    return c



def FastNorm():
    m_Scale = 30000000.0
    m_Rscale = 1.0 / m_Scale
    m_Rcons = 1.0 / (2.0 * 1024.0 * 1024.0 * 1024.0)
    m_ELEN = 7  #LEN must be 2 ** ELEN  
    m_LEN = 128
    m_LMASK = (4 * (m_LEN - 1))
    m_TLEN = (8 * m_LEN)
    m_Vec1 = np.zeros(m_TLEN)
    m_Lseed = 12345
    m_Irs = 12345
    m_GScale = m_Rscale
    fake = 1.0 + 0.125 / m_TLEN
    m_Chic2 = np.sqrt(2.0 * m_TLEN - fake * fake) / fake
    m_Chic1 = fake * np.sqrt(0.5 / m_TLEN)
    m_ActualRSD = 0.0
    inc = 0
    mask = 0
    m_Nslew = 0
    if (not(m_Nslew & 0xFF)):
        if (m_Nslew & 0xFFFF):
            print('go to recalcsumsq')
        else:
            ts = 0.0
            p = 0
            while(True):
                while(True):
                    m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                    m_Irs = np.int64(SignedShiftXOR(m_Irs))
                    r = np.int32((m_Irs)+ np.int64(m_Lseed))
                    tx = m_Rcons * r
                    m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                    m_Irs = np.int64(SignedShiftXOR(m_Irs))
                    r = np.int32((m_Irs) + np.int64(m_Lseed))
                    ty = m_Rcons * r
                    tr = tx * tx + ty * ty
                    if ((tr <= 1.0) and (tr >= 0.1)):
                        break
                m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                m_Irs = np.int64(SignedShiftXOR(m_Irs))
                r = np.int32((m_Irs) + np.int64(m_Lseed))
                if (r < 0):
                    r = ~r
                tz = -2.0 * np.log((r + 0.5) * m_Rcons)
                ts += tz
                tz = np.sqrt(tz / tr)
                m_Vec1[p] = (int)(m_Scale * tx * tz)
                p=p+1
                m_Vec1[p] = (int)(m_Scale * ty * tz)
                p=p+1
                if (p >= m_TLEN):
                    break
            ts = m_TLEN / ts
            tr = np.sqrt(ts)
            for p in range(0, m_TLEN):
                tx = m_Vec1[p] * tr
                m_Vec1[p]= int(tx - 0.5) if int(tx < 0.0) else int(tx + 0.5)
            ts = 0.0
            for p in range(0,m_TLEN):
                tx = m_Vec1[p]
                ts += (tx * tx)
            ts = np.sqrt(ts / (m_Scale * m_Scale * m_TLEN))
            m_ActualRSD = 1.0 / ts
            m_Nslew=m_Nslew+1
            global m_Gaussfaze
            m_Gaussfaze = m_TLEN - 1
            m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
            m_Irs = np.int64(SignedShiftXOR(m_Irs))
            t = np.int32((m_Irs) + np.int64(m_Lseed))
            if (t < 0):
                t = ~t
            t = t >> (29 - 2 * m_ELEN)
            skew = (m_LEN - 1) & t
            t = t >> m_ELEN
            skew = 4 * skew
            stride = int((m_LEN / 2 - 1)) & t
            t = t >> (m_ELEN - 1)
            stride = 8 * stride + 4
            mtype = t & 3
            stype = m_Nslew & 3
            if(stype==1):
                inc = 1
                mask = m_LMASK
                pa = m_Vec1[4 * m_LEN]
                pa_idx = 4 * m_LEN
                pb = m_Vec1[4 * m_LEN + m_LEN]
                pb_idx = 4 * m_LEN + m_LEN
                pc = m_Vec1[4 * m_LEN + 2 * m_LEN]
                pc_idx = 4 * m_LEN + 2 * m_LEN
                pd = m_Vec1[4 * m_LEN + 3 * m_LEN]
                pd_idx = 4 * m_LEN + 3 * m_LEN
                p0 = m_Vec1[0]
                p0_idx = 0
                global m_Gausssave
                m_Gausssave = m_Vec1
                i = m_LEN
                pb = m_Vec1[4 * m_LEN + m_LEN + (inc * (m_LEN - 1))]
                pb_idx = 4 * m_LEN + m_LEN + (inc * (m_LEN - 1))
                while(True):
                    skew = (skew + stride) & mask
                    pe = m_Vec1[skew]
                    pe_idx = skew
                    p = -m_Vec1[pa_idx]
                    q = m_Vec1[pb_idx]
                    r = m_Vec1[pc_idx]
                    s = -m_Vec1[pd_idx]
                    t = int(p + q + r + s) >> 1
                    p = t - p
                    q = t - q
                    r = t - r
                    s = t - s
  
                    t = m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = p
                    pe = m_Vec1[skew+inc]
                    pe_idx = skew+inc
                    p = -m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = q
                    pe = m_Vec1[skew + 2 * inc]
                    pe_idx = skew + 2 * inc
                    q = -m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = r
                    pe = m_Vec1[skew + 3 * inc]
                    pe_idx = skew + 3 * inc
                    r = m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = s
                    s = int(p + q + r + t) >> 1
                    m_Vec1[pa_idx] = s - p
                    pa = m_Vec1[pa_idx + inc]
                    pa_idx = pa_idx + inc
                    m_Vec1[pb_idx] = s - t
                    pb = m_Vec1[pb_idx - inc]
                    pb_idx = pb_idx - inc
                    m_Vec1[pc_idx] = s - q
                    pc = m_Vec1[pc_idx + inc]
                    pc_idx = pc_idx + inc
                    m_Vec1[pd_idx] = s - r
                    if(i==1):
                        break
                    else:
                        pd = m_Vec1[pd_idx + inc] 
                        pd_idx = pd_idx + inc
                    i=i-1
                    if (i==0):
                        break
                ts = m_Chic1 * (m_Chic2 + m_GScale * m_Vec1[m_TLEN - 1])
                m_GScale = m_Rscale * ts * m_ActualRSD
                return (m_GScale * m_Vec1[0])
  
            else:
                print('ERRORE')
    else:
        return 10


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

def OnePlusOne(Ref_uint8, Flt_uint8, volume):
    
    m_CatchGetValueException = False
    m_MetricWorstPossibleValue = 0

    m_Maximize = False
    m_Epsilon = 1.5e-4

    m_Initialized = False
    m_GrowthFactor = 1.05
    m_ShrinkFactor = np.power(m_GrowthFactor, -0.25)
    m_InitialRadius = 1.01
    m_MaximumIteration = 200
    m_Stop = False
    m_CurrentCost = 0
    m_CurrentIteration = 0
    m_FrobeniusNorm = 0.0

    spaceDimension = 6

    A = np.identity(spaceDimension)*(m_InitialRadius) 
    parent = np.zeros((3,4))
    estimate_initial(Ref_uint8, Flt_uint8, parent, volume) 
    Ref_uint8 = torch.ravel(Ref_uint8)
    f_norm = np.zeros(spaceDimension)
    child = np.array(spaceDimension)
    delta = np.array(spaceDimension)
    
    parentPosition = np.array(spaceDimension)
    childPosition = np.array(spaceDimension)

    parentPosition=[parent[0][3],parent[1][3],0, 1.0, 1.0, parent[0][0]]
    transformed = transform(Flt_uint8, parent, volume)
    transformed = torch.from_numpy(transformed)
    pvalue= mutual_info_sw(volume, Ref_uint8, transformed, 512)*-1

    m_CurrentIteration = 0
    
    for i in range (0,m_MaximumIteration):
        m_CurrentIteration=m_CurrentIteration+1
    
        for j in range (0, spaceDimension):
            f_norm[j]= NormalVariateGenerator() 
    
        delta = A.dot(f_norm)#A * f_norm
        child = parentPosition + delta
        childPosition = to_matrix_complete(child)
        transformed2 = transform(Flt_uint8, childPosition, volume)
        transformed2 = torch.from_numpy(transformed2)
        cvalue= mutual_info_sw(volume, Ref_uint8, transformed2, 512)*-1

        adjust = m_ShrinkFactor
    
        if(m_Maximize):
            if(cvalue>pvalue):
                pvalue = cvalue
                child, parentPosition = parentPosition, child 
                adjust = m_GrowthFactor
            else:
                pass
        else:
            if(cvalue < pvalue):
                pvalue = cvalue
                child, parentPosition = parentPosition, child 
                adjust = m_GrowthFactor
            else:
                pass
                
            
        m_CurrentCost = pvalue
        m_FrobeniusNorm = np.linalg.norm(A,'fro')
    
        if(m_FrobeniusNorm <= m_Epsilon):
            break
    
        alpha = ((adjust - 1.0) / np.dot(f_norm, f_norm))
    
        for c in range(0, spaceDimension):
            for r in range(0,spaceDimension):
                A[r][c] += alpha * delta[r] * f_norm[c]

    return (parentPosition)  

def register_images(Ref_uint8, Flt_uint8, volume, filename):
    start_single_sw = time.time()

    optimal_params = OnePlusOne(Ref_uint8, Flt_uint8,  volume) 
    params_trans=to_matrix_complete(optimal_params)
    flt_transform = transform(Flt_uint8, params_trans, volume)
    end_single_sw = time.time()
    print('Final time: ', end_single_sw - start_single_sw)
    with open(filename, 'a') as file2:
        file2.write("%s\n" % (end_single_sw - start_single_sw))
 
    return (params_trans)


def compute_wrapper(args, hephaestus_pl):
    num_threads = 1
    config=args.config
    the_list_of_wonderful_lists=hephaestus.hephaestus_accel_map(programmable_logic=hephaestus_pl, \
        number_of_cores=num_threads, \
        image_dimension=args.image_dimension, volume = args.volume, platform=args.platform, \
        )
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

def compute(CT, PET, volume, filename, curr_res, t_id, patient_id, wonderful_list, first_slice, last_slice, num_subvolumes):
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
    compute_wrapper(args,num_threads)

    print("hephaestus (1+1) python is at the end :)")                                
        


if __name__== "__main__":
    main()
