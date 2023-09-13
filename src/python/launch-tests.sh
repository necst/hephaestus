# /******************************************
# *MIT License
# *
# *Copyright (c) [2023] [Giuseppe Sorrentino, Marco Venere, Davide Conficconi, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *Copyright (c) [2022] [Eleonora D'Arnese, Davide Conficconi, Emanuele Del Sozzo, Luigi Fusco, Donatella Sciuto, Marco Domenico Santambrogio]
# *Copyright (c) [2020] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Donatella Sciuto, Marco D. Santambrogio]

# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# ******************************************/


#Uncomment the set of commands you want to execute, please remember to modify paths with your absolute paths

source /opt/xilinx/xrt/setup.sh
# run test mutual information with one or two cores
#python3 src/python/test_mutual_information.py -nc 1 -ol ../new_build_hephaestus/1core-16pe-16-clock-300/mutual_information_master.xclbin
#python3 src/python/test_mutual_information.py -nc 2 -ol ../new_build_hephaestus/2core-8pe-16-clock-300/mutual_information_master.xclbin

#Run this 3 commands to execute One Plus One, volume = 246, computing the accuracy.
# python3 src/python/hephaestus-oneplusone-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/output1p1_nop8/ -ol ../new_build_hephaestus/1core-16pe-16-clock-300/mutual_information_master.xclbin -im 512 -vol 246 -f Time1p1_u200_8pe_cache_noprint.csv
# python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/output1p1_nop8/ -l i1p1_u200_8pe_8pen_maybe_cache_noprint -rp ./
# python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-i1p1_u200_8pe_8pen_maybe_cache_noprint-score_results.csv

#Run this 3 commands to execute One Plus One, choosing subvolume with left and right index, computing the accuracy. In this example we examine from slice 120 to slice 180
#python3 src/python/hephaestus-oneplusone-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/output1p1_nop8/ -ol ../new_build_hephaestus/1core-16pe-16-clock-300/mutual_information_master.xclbin -im 512 -fs 120 -ls 180 -vol 246 -f Time1p1_u200_8pe_cache_noprint.csv -as False
#python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/output1p1_nop8/ -l i1p1_u200_8pe_8pen_maybe_cache_noprint -rp ./
#python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-i1p1_u200_8pe_8pen_maybe_cache_noprint-score_results.csv

#Run this 3 commands to execute Powell 1, volume = 246, computing the accuracy.
# python3 src/python/hephaestus-powell-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -ol ../new_build_hephaestus/1core-16pe-16-clock-300/mutual_information_master.xclbin -nc 1 -im 512 -vol 246 -f Timepow_u200_8pe_cache_noprint.csv
# python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -l ipow_u200_8pe_8pen_maybe_cache_noprint -rp ./
# python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-ipow_u200_8pe_8pen_maybe_cache_noprint-score_results.csv


# #Run this 3 commands to execute Powell 2, volume = 246, computing the accuracy.
# python3 src/python/hephaestus-powell-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -ol ../new_build_hephaestus/2core-8pe-16-clock-300/mutual_information_master.xclbin -nc 2 -im 512 -vol 246 -f Timepow_u200_8pe_cache_noprint.csv 
# python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -l ipow_u200_8pe_8pen_maybe_cache_noprint -rp ./
# python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-ipow_u200_8pe_8pen_maybe_cache_noprint-score_results.csv


#Run this 3 commands to execute Powell 1 core, choosing subvolume with left and right index, computing the accuracy. In this example we examine from slice 120 to slice 180
# python3 src/python/hephaestus-powell-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -ol ../new_build_hephaestus/1core-16pe-16-clock-300/mutual_information_master.xclbin -im 512 -nc 1 -fs 120 -ls 180 -vol 246 -f Timepow_u200_8pe_cache_noprint.csv -as False
# python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -l ipow_u200_8pe_8pen_maybe_cache_noprint -rp ./
# python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-ipow_u200_8pe_8pen_maybe_cache_noprint-score_results.csv


#Run this 3 commands to execute Powell 2 core, choosing subvolume with left and right index, computing the accuracy. In this example we examine from slice 120 to slice 180
# python3 src/python/hephaestus-powell-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -ol ../new_build_hephaestus/2core-8pe-16-clock-300/mutual_information_master.xclbin -im 512 -nc 2 -fs 120 -ls 180 -vol 246 -f Timepow_u200_8pe_cache_noprint.csv -as False
# python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -l ipow_u200_8pe_8pen_maybe_cache_noprint -rp ./
# python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-ipow_u200_8pe_8pen_maybe_cache_noprint-score_results.csv

#Run these 3 commands to execute OnePlusOne 1 core on Zynq, with automatic sizing of the subvolume for searching registration parameters:
#python3 src/python/hephaestus-oneplusone-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/output1p1_nop8/ -ol ../new_build_hephaestus/1core-4pe-16-clock-150/iron_wrapper.bit -im 512 -vol 246 -f Time1p1_zcu_cache_noprint.csv -p Zynq
#python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/output1p1_nop8/ -l i1p1_zcu -rp ./
#python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-i1p1_zcu-score_results.csv


#Run these 3 commands to execute Powell 1 core on Zynq, with automatic sizing of the subvolume for searching registration parameters:
#python3 src/python/hephaestus-powell-complete.py -cp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE0 -pp /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/SE4 -rp /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -ol ../new_build_hephaestus/1core-4pe-16-clock-150/iron_wrapper.bit -im 512 -vol 246 -f TimePow_zcu_cache_noprint.csv -p Zynq
#python3 src/scripts/res_extraction.py -f 0 -rg /home/users/giuseppe.sorrentino/Hephaestus/dataset/Test/ST0/NuovoGold/ -rt /home/users/giuseppe.sorrentino/Hephaestus/src/dataset/outputpow_nop8/ -l ipow_zcu -rp ./
#python3 src/scripts/AVGcompute.py -f /home/users/giuseppe.sorrentino/Hephaestus/gold-ipow_zcu-score_results.csv
