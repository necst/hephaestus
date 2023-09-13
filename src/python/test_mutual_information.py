#!/usr/bin/env python
# coding: utf-8
#/******************************************
#*MIT License
#*
# *Copyright (c) [2023] [Giuseppe Sorrentino, Marco Venere, Davide Conficconi, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *Copyright (c) [2022] [Eleonora D'Arnese, Davide Conficconi, Emanuele Del Sozzo, Luigi Fusco, Donatella Sciuto, Marco Domenico Santambrogio]
# *Copyright (c) [2020] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Donatella Sciuto, Marco D. Santambrogio]

#*
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

from pynq import Device
from pynq.pmbus import DataRecorder
import os
import cv2
import numpy as np
import math
import glob
import time
import pandas as pd
import multiprocessing
from pynq import Overlay
import pynq
from pynq import allocate
import struct
import statistics
import argparse


import pynq
from pynq import allocate




# function for specific multicore mapping on different platforms, memory banks and namings
def mi_accel_map(hephaestus_pl, platform, i_ref_sz=512, config=None, n_couples = 1, board = None, num_cores = 1):
    if platform == "Alveo":
        if board == "alveo_u200":
            bank0 = hephaestus_pl.bank0
            bank1 = hephaestus_pl.bank1
        elif board == "alveo_u280" and num_cores == 1:
            bank0 = hephaestus_pl.DDR0
            bank1 = hephaestus_pl.DDR1
        elif board == "alveo_u280" and num_cores == 2:
            bank0 = hephaestus_pl.DDR0
            bank1 = hephaestus_pl.DDR0
    mi_list = []
    ref_size=i_ref_sz
    ref_dt="uint8"
    flt_size=i_ref_sz
    flt_dt="uint8"
    mi_size=1
    mi_dt=np.float32


    if platform == 'Alveo':#pcie card based
        mi_acc_0=SingleAccelMI(hephaestus_pl.mutual_information_master_1_1, platform, [bank0, bank1], ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples)
    else: #ZYNQ based
        mi_acc_0=SingleAccelMI(hephaestus_pl.mutual_information_m_0, platform, None, ref_size, ref_dt, flt_size, flt_dt, mi_size, mi_dt, n_couples, config)
    mi_list.append(mi_acc_0)
    return mi_list





class SingleAccelMI :
    
    
###########################################################
# DEFAULTS of the INIT
###########################################################
#
# platform='Alveo'
#caching=False
#ref_size=512
# ref_dt="uint8"
# flt_size=512, then to the power of 2
#flt_dt="uint8"
# mi_size=1 then to the power of 2
# n_couples = 1
# mi_dt=np.float32
#
###########################################################

    def __init__(self, accel_id,  platform='Alveo', mem_bank=None, ref_size=512, ref_dt="uint8", flt_size=512, flt_dt="uint8", mi_size=1, mi_dt=np.float32, n_couples = 1, config=None):
            self.AP_CTRL = 0x00
            self.done_rdy = 0x6
            self.ap_start = 0x1
            self.REF_ADDR = 0x18
            self.FLT_ADDR_OR_MI = 0x10
            self.MI_ADDR_OR_FUNCT = 0x20
            self.N_COUPLES_ADDR =0x28
            
            self.LOAD_IMG = 0
            self.COMPUTE = 1
            self.n_couples = n_couples
            if platform == "Alveo":
                self.buff1_img = allocate(n_couples*ref_size*ref_size, ref_dt, target=mem_bank[0])
                self.buff2_img_mi = allocate(n_couples*flt_size*flt_size, flt_dt, target=mem_bank[1])
                self.buff3_mi_status = allocate(mi_size, mi_dt, target=mem_bank[1])
            else:
                self.buff1_img = allocate(n_couples*ref_size*ref_size, ref_dt, target=mem_bank)
                self.buff2_img_mi = allocate(n_couples*flt_size*flt_size, flt_dt, target=mem_bank)
                self.buff3_mi_status = allocate(mi_size, mi_dt, target=mem_bank)

            self.buff1_img_addr = self.buff1_img.device_address
            self.buff2_img_mi_addr = self.buff2_img_mi.device_address
            self.buff3_mi_status_addr = self.buff3_mi_status.device_address
            
            self.accel = accel_id
            
            self.platform = platform
            self.config = config

    def get_config(self):
        return self.config

    def init_accel(self, Ref_uint8, Flt_uint8):
        self.prepare_ref_buff(Ref_uint8)
        self.prepare_flt_buff(Flt_uint8)
    
    def read_status(self):
        return self.accel.mmio.read(self.STATUS_ADDR)

    def prepare_ref_buff(self, Ref_uint8):
        #self.buff1_img[:] = Ref_uint8.flatten()7
        data = Ref_uint8.flatten()
        self.buff1_img[:] = 0
        self.buff1_img[0:len(data)] = data
        self.buff1_img.flush()#sync_to_device
        return
    
    def prepare_flt_buff(self, Flt_uint8):
        #self.buff2_img_mi[:] = Flt_uint8.flatten()
        data = Flt_uint8.flatten()
        self.buff2_img_mi[:] = 0
        self.buff2_img_mi[0:len(data)] = data[:]
        self.buff2_img_mi.flush() #sync_to_device
            
    def set_n_couples(self, n):
        self.n_couples = n
        
    def execute_zynq(self, mi_addr_or_funct):
        self.accel.write(self.REF_ADDR, self.buff1_img.device_address)
        self.accel.write(self.FLT_ADDR_OR_MI, self.buff2_img_mi.device_address)
        self.accel.write(self.MI_ADDR_OR_FUNCT, mi_addr_or_funct)
        self.accel.write(self.N_COUPLES_ADDR, self.n_couples)
        self.accel.write(self.AP_CTRL, self.ap_start)
        while(self.accel.mmio.read(0) & 0x4 != 0x4):
            pass
    
    def exec_and_wait(self):
        result = []
        if self.platform == 'Alveo':
            self.accel.call(self.buff1_img, self.buff2_img_mi, self.buff3_mi_status, self.n_couples)
        else:# ZYNQ based
            self.execute_zynq(self.buff3_mi_status.device_address)
        self.buff3_mi_status.invalidate()#sync_from_device
        result.append(self.buff3_mi_status)
    
        return result

    def zero_cma_buff(self):
        self.buff1_img[:] = 0
        self.buff2_img_mi[:] = 0
        
    def reset_cma_buff(self):
	
        self.buff1_img.freebuffer() 
        self.buff2_img_mi.freebuffer()
        self.buff3_mi_status.freebuffer()
        del self.buff1_img 
        del self.buff2_img_mi
        del self.buff3_mi_status
    
    def mutual_info_sw(self, Ref_uint8, Flt_uint8, dim):
        j_h=np.histogram2d(Ref_uint8.ravel(),Flt_uint8.ravel(),bins=[256,256])[0]
        j_h=j_h/(self.n_couples*dim*dim)
          
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



def main():
    
    parser = argparse.ArgumentParser(description='Hephaestus software for IR onto a python env')
    parser.add_argument("-b", "--board", nargs='?', help='Board Code Name', default='alveo_u280')
    parser.add_argument("-nc", "--num_cores", nargs='?', help='Number of // cores', default=1, type=int)
    parser.add_argument("-ol", "--overlay", nargs='?', help='Path and filename of the target overlay', default='')
    parser.add_argument("-clk", "--clock", nargs='?', help='Target clock frequency of the PL', default=100, type=int)
    parser.add_argument("-p", "--platform", nargs='?', help='platform to target.\
     \'Alveo\' is used for PCIe/XRT based,\n while \'Ultra96\' will setup for a Zynq-based environment', default='Alveo')
    parser.add_argument("-im", "--image_dimension", nargs='?', help='Target images dimensions', default=512, type=int)
    #parser.add_argument("-rp", "--res_path", nargs='?', help='Path of the Results', default='./')
    parser.add_argument("-c", "--config", nargs='?', help='hw config to print only', default='ok')
    parser.add_argument("-f", "--filename", nargs='?', help='Name of the file in which to write times', default='test.csv')
    #parser.add_argument("-nc", "--n_couples", help="sets the positive number of couples of ref and flt passed, default 1", default='1', type=int)
    
    #overlay = "~/alveo_build/8pezerofunzionanti/mutual_information_master.xclbin"
    args = parser.parse_args()
    hephaestus1 = Overlay(args.overlay)
    values = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,50,100,150,200,250,300,350,400,450,500,512]
    for seed in (1234, 0, 98562,):
        #for n_couples in [1,50,100,150,200,250,300,350,400,450,500,512]:
        for n_couples in values:

            hist_dim = 256
            dim = 512
            t=0
            #args = parser.parse_args()
            
            clock = args.clock
            accel_number = 1 
            platform = args.platform
            image_dimension = args.image_dimension
            res_path = "./"
            config = args.config
            board = args.board

            if platform=='Zynq':
                from pynq.ps import Clocks;
                Clocks.fclk0_mhz = clock; 
            np.random.seed(seed)
            
            accel_list=mi_accel_map(hephaestus1, platform, image_dimension, config, n_couples, board, args.num_cores)

            
            iterations=10
            t_tot = 0
            times=[]
            time_sw = []
            time_hw = []
            dim=image_dimension
            diffs=[]
            hw_mi=[]
            print('seed: %d, ncouples: %d' % (seed, n_couples))
            for i in range(iterations):
                ref = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
                if seed in (1234, 0, 98562):
                    flt = np.random.randint(low=0, high=255, size = (n_couples*image_dimension, image_dimension), dtype='uint8')
                elif seed in (73541, 3478, 87632):
                    flt = ref
                else:
                    flt = np.zeros((n_couples*image_dimension, image_dimension))
                start_single_sw = time.time()
                sw_mi=accel_list[0].mutual_info_sw(ref, flt, dim)
                end_single_sw = time.time()
                time_sw.append(end_single_sw - start_single_sw)
                accel_list[0].prepare_ref_buff(ref)
                accel_list[0].prepare_flt_buff(flt)
                start_single = time.time()
                out = accel_list[0].exec_and_wait()
                hw_mi.append(out)
                end_single = time.time()
                t = end_single - start_single
                times.append(t)
                time_hw.append(t)
                diff=sw_mi - out[0]
                diffs.append(diff)
                t_tot = t_tot +  t
            end_tot = time.time()
            accel_list[0].reset_cma_buff()
            with open(args.filename, 'a') as file:
                ratio = np.divide(time_sw, time_hw)
                file.write("float, %d, 32, %d, %s, %s, %s, %s, %s,%s,\n" % (n_couples, seed, np.mean(times), np.std(times), np.mean(time_sw), np.std(time_sw), np.mean(diffs),np.std(diffs),))        
    hephaestus1.free()
if __name__== "__main__":
    main()