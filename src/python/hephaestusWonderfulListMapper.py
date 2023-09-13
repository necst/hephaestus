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
import pynq
import numpy as np
from hephaestusNewFunctions import *
import hephaestusNewFunctions

def hephaestus_accel_map(programmable_logic=None, number_of_cores=1, image_dimension=512, volume=1, platform='Alveo', board=None):
        return mi_map(programmable_logic, number_of_cores,image_dimension,volume, platform, board)


def append_useless(wonderful_list,number_of_useless_items):
    for i in range(number_of_useless_items):
        wonderful_list.append(str(i))
    return wonderful_list


def mi_map(programmable_logic=None, number_of_cores=1, image_dimension=512, volume=1, platform='Alveo', board=None):
    the_list_of_wonderful_lists=[]
    wonderful_list=[]
    mem_bank=None
    if platform == "Alveo":
        if board == "alveo_u200":
            bank0 = programmable_logic.bank0
            bank1 = programmable_logic.bank1
        elif board == "alveo_u280":
            bank0 = programmable_logic.DDR0
            bank1 = programmable_logic.DDR1
    if number_of_cores >= 1 and (platform == "Zynq" or (platform == "Alveo" and board == "alveo_u200")):
        wonderful_list.append(hephaestusNewFunctions.compute_hw_sm_negative)
        
        if platform =='Alveo':
            wonderful_list.append(programmable_logic.mutual_information_master_1_1)
            mem_bank=[bank0, bank1]
        else:
            wonderful_list.append(programmable_logic.mutual_information_m_0)
        hephaestusNewFunctions.fill_wonderful_list_std_alone(size_single=image_dimension, \
            volume = volume, hephaestus_pl=programmable_logic, mem_bank=mem_bank,\
            wonderful_list=wonderful_list, platform=platform)
        append_useless(wonderful_list,1)
        wonderful_list.append(platform)
        append_useless(wonderful_list,4)
    
    elif number_of_cores == 1 and platform == "Alveo" and board == "alveo_u280" :
        wonderful_list.append(hephaestusNewFunctions.compute_hw_sm_negative)
        
        if platform =='Alveo':
            wonderful_list.append(programmable_logic.mutual_information_master_1_1)
            mem_bank=[bank0, bank1]
        else:
            wonderful_list.append(programmable_logic.mutual_information_m_0)
        hephaestusNewFunctions.fill_wonderful_list_std_alone(size_single=image_dimension, \
            volume = volume, hephaestus_pl=programmable_logic, mem_bank=mem_bank,\
            wonderful_list=wonderful_list, platform=platform)
        
        append_useless(wonderful_list,1)
        wonderful_list.append(platform)
        append_useless(wonderful_list,4)
    elif number_of_cores >= 1 and platform == "Alveo" and board == "alveo_u280":
        wonderful_list.append(hephaestusNewFunctions.compute_hw_sm_negative)
        
        if platform =='Alveo':
            wonderful_list.append(programmable_logic.mutual_information_master_1_1)
            mem_bank=[bank0, bank0]
        else:
            wonderful_list.append(programmable_logic.mutual_information_m_0)
        hephaestusNewFunctions.fill_wonderful_list_std_alone(size_single=image_dimension, \
            volume = volume, hephaestus_pl=programmable_logic, mem_bank=mem_bank,\
            wonderful_list=wonderful_list, platform=platform)
        append_useless(wonderful_list,1)
        wonderful_list.append(platform)
        append_useless(wonderful_list,4)

    if number_of_cores >=2 and (platform == "Zynq" or (platform == "Alveo" and board == "alveo_u200")):
        the_list_of_wonderful_lists.append(wonderful_list)
        wonderful_list=[]
        wonderful_list.append(hephaestusNewFunctions.compute_hw_sm_negative)
        if platform =='Alveo':
            wonderful_list.append(programmable_logic.mutual_information_master_2_1)
            mem_bank=[programmable_logic.bank2, programmable_logic.bank3] 
        else:
            wonderful_list.append(programmable_logic.mutual_information_m_1)
        hephaestusNewFunctions.fill_wonderful_list_std_alone(size_single=image_dimension, \
            volume = volume, hephaestus_pl=programmable_logic, mem_bank=mem_bank,\
            wonderful_list=wonderful_list,platform=platform)
        append_useless(wonderful_list,1)
        wonderful_list.append(platform)
        append_useless(wonderful_list,4)

    elif number_of_cores >=2 and platform == "Alveo" and board == "alveo_u280":
        the_list_of_wonderful_lists.append(wonderful_list)
        wonderful_list=[]
        wonderful_list.append(hephaestusNewFunctions.compute_hw_sm_negative)
        if platform =='Alveo':
            wonderful_list.append(programmable_logic.mutual_information_master_2_1)
            mem_bank=[programmable_logic.DDR1, programmable_logic.DDR1] 
        else:
            wonderful_list.append(programmable_logic.mutual_information_m_1)
        hephaestusNewFunctions.fill_wonderful_list_std_alone(size_single=image_dimension, \
            volume = volume, hephaestus_pl=programmable_logic, mem_bank=mem_bank,\
            wonderful_list=wonderful_list,platform = platform)
        append_useless(wonderful_list,1)
        wonderful_list.append(platform)
        append_useless(wonderful_list,4)



    
    the_list_of_wonderful_lists.append(wonderful_list)
    return the_list_of_wonderful_lists


