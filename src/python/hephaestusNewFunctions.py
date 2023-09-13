#!/usr/bin/env python
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
import numpy as np
import cv2
import pynq
from pynq import allocate
import numpy as np
import cv2
import struct
import hephaestusImageRegistration as fir

AP_CTRL = 0x00
done_rdy = 0x6
ap_start = 0x1
AP_REF_ADDR = 0x18
AP_FLT_ADDR_OR_MI = 0x10
AP_MI_ADDR_OR_FUNCT = 0x20
AP_N_COUPLES_ADDR = 0x28

def execute_zynq(target_accel):
    target_accel.write(AP_CTRL, ap_start)
    while(target_accel.mmio.read(0) & 0x4 != 0x4):
        pass

def fill_buff(my_buffer,data):
    my_buffer[:] = 0
    d = data.ravel()
    my_buffer[:len(d)] = d
    my_buffer.flush()

def fill_two_buff(my_buffer1,my_buffer2,data1,data2):
    my_buffer1[:] = 0
    my_buffer2[:] = 0
    d1 = data1.ravel()
    d2 = data2.ravel()
    my_buffer1[:len(d1)] = d1
    my_buffer2[:len(d2)] = d2
    my_buffer1.flush()
    my_buffer2.flush()

def fill_three_buff(my_buffer1,my_buffer2,my_buffer3,data1,data2,data3):
    my_buffer1[:] = 0
    my_buffer2[:] = 0
    my_buffer3[:] = 0
    d1 = data1.ravel()
    d2 = data2.ravel()
    d3 = data3.ravel()
    my_buffer1[:len(d1)] = d1
    my_buffer2[:len(d2)] = d2
    my_buffer3[:len(d3)] = d3
    my_buffer1[:] = data1.ravel()
    my_buffer2[:] = data2.ravel()
    my_buffer3[:] = data3.ravel()
    my_buffer1.flush()
    my_buffer2.flush()
    my_buffer3.flush()

def read_buff(my_buffer, my_list):
    my_buffer.invalidate()
    my_list.append(my_buffer)

def write_axilite(target_accel,trgt_addres,trgt_data):
    target_accel.write(trgt_addres, trgt_data)

def execute_signature_zynq_std_alone(target_accel,in1_vadd,in2_vadd,out0,in3_vadd):
    write_axilite(target_accel,AP_REF_ADDR,in1_vadd.device_address)
    write_axilite(target_accel,AP_FLT_ADDR_OR_MI,in2_vadd.device_address)
    write_axilite(target_accel, AP_N_COUPLES_ADDR, in3_vadd)
    write_axilite(target_accel,AP_MI_ADDR_OR_FUNCT,out0.device_address)
    execute_zynq(target_accel)


def compute_hw_sm_negative(target_accel,ref,flt,volume,params,in1_vadd,in2_vadd,in3_vadd,out0,useless,platform,useless2,useless3,useless4):
    transformed = fir.transform(flt, params, volume)
    fill_buff(in2_vadd,transformed)

    if platform != "Zynq":
        target_accel.call(in1_vadd,in2_vadd,out0,volume)
    else:
        execute_signature_zynq_std_alone(target_accel,in1_vadd,in2_vadd,out0, volume)
    
    out0.invalidate()
    data = struct.pack('d',out0)
    val=struct.unpack('d',data)
    mi = val[0]
    return -mi

def compute_hw_sm_negative_dc(target_accel1, target_accel2,ref,transformed1, transformed2,volume,params,ref1_vadd,flt1_vadd,ref2_vadd, flt2_vadd, in3_vadd,out1,out2,useless,platform,useless2,useless3,useless4):
  
    fill_buff(flt1_vadd,transformed1)
    fill_buff(flt2_vadd,transformed2)

    

    if platform != "Zynq":
        w1 = target_accel1.start(ref1_vadd,flt1_vadd,out1, volume)
        w2 = target_accel2.start(ref2_vadd,flt2_vadd,out2, volume)
        w2.wait()
    else:
        execute_signature_zynq_std_alone(target_accel1,ref1_vadd,flt1_vadd,out1, volume)
    
    out1.invalidate()
    
    data1 = struct.pack('d',out1)
    val1=struct.unpack('d',data1)
    mi_a = val1[0]

    out2.invalidate()
    
    data2 = struct.pack('d',out2)
    val2=struct.unpack('d',data2)
    mi_b = val2[0]
    return (-mi_a, -mi_b)

def fill_wonderful_list_std_alone(size_single=512, volume=1,hephaestus_pl=None, mem_bank=None,wonderful_list=None, platform = 'Alveo'):
    
    if platform == 'Zynq':
        in1_vadd = pynq.allocate((size_single * size_single * volume), np.uint8,mem_bank)
        in2_vadd = pynq.allocate((size_single * size_single * volume), np.uint8,mem_bank)
        in3_vadd = volume
        out0 = pynq.allocate((1), np.float32,mem_bank)
        wonderful_list.append(in1_vadd)
        wonderful_list.append(in2_vadd)
        wonderful_list.append(volume)
        wonderful_list.append(out0)
    else:
        in1_vadd = pynq.allocate((size_single * size_single * volume), np.uint8,mem_bank[0])
        in2_vadd = pynq.allocate((size_single * size_single * volume), np.uint8,mem_bank[1])
        in3_vadd = volume
        out0 = pynq.allocate((1), np.float32,mem_bank[1])
        wonderful_list.append(in1_vadd)
        wonderful_list.append(in2_vadd)
        wonderful_list.append(volume)
        wonderful_list.append(out0)

    

