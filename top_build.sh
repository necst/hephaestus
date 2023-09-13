#!/bin/bash
# /******************************************
# *MIT License
# *
# # *Copyright (c) [2023] [Giuseppe Sorrentino, Marco Venere, Davide Conficconi, Eleonora D'Arnese, Marco Domenico Santambrogio]
# # *Copyright (c) [2022] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Donatella Sciuto, Marco D. Santambrogio]
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
# */


###############TOP configuration generation###################
#PYNQZU
make hw_gen PE=2 CORE_NR=1 TARGET=hw OPT_LVL=3 PE_ENTROP=16 FREQ_MHZ=217 TRGT_PLATFORM=pynqzu NCM=512

#Zcu104
make hw_gen PE=4 CORE_NR=1 TARGET=hw OPT_LVL=3 PE_ENTROP=16 FREQ_MHZ=215 TRGT_PLATFORM=zcu104 NCM=512

#Alveo u200
make hw_gen PE=8 CORE_NR=1 TARGET=hw OPT_LVL=3 CLK_FRQ=300 PE_ENTROP=8 TRGT_PLATFORM=alveo_u200 NCM=512
make hw_gen PE=8 CORE_NR=2 TARGET=hw OPT_LVL=3 CLK_FRQ=300 PE_ENTROP=8 TRGT_PLATFORM=alveo_u200 NCM=512

#Alveo u280
make hw_gen PE=16 CORE_NR=1 TARGET=hw OPT_LVL=3 CLK_FRQ=300 PE_ENTROP=16 TRGT_PLATFORM=alveo_u280 NCM=512
make hw_gen PE=8 CORE_NR=2 TARGET=hw OPT_LVL=3 CLK_FRQ=300 PE_ENTROP=16 TRGT_PLATFORM=alveo_u280 NCM=512