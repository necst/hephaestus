#!/bin/bash
# /******************************************
# *MIT License
# *
# *Copyright (c) [2023] [Giuseppe Sorrentino, Marco Venere, Davide Conficconi, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *Copyright (c) [2022] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Donatella Sciuto, Marco Domenico Santambrogio]

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

PES=( 1 2 4 8 16 32)
NCM=( 8 16 32 64 128 256 512)
CORE_NRS=(1 2)
PE_ENTROP=(1 2 4 8 16 32) 
FREQZ=300
#this value affect vivado design frequency in all platforms, not in HLS standalone
#change this value if using a zynq-based, use 200MHz / 250MHz / 300MHz
#change this value if using an alveo-based, use 300MHz 


#Change TRGT_PLATFORM=`pynqz2|ultra96_v2|zcu104|alveo_u200` to target different platforms

echo "************************"
echo "************************"
echo "************************"
echo "************************"
echo ""
echo "[WARNING] This script generates M*O^N designs where N is the variety of parameters"
echo " (i.e., for depth) and M the different possibilities of each parameter"
echo ""
echo "************************"
echo "************************"
echo "************************"
echo "************************"

for p in ${PES[@]}; do
    for ncm in ${NCM[@]}; do
        for cn in ${CORE_NRS[@]}; do
            for pe in ${PE_ENTROP[@]}; do
                for f in ${FREQ[@]}; do
                        #make hw_gen PE=$p CORE_NR=$cn HT=$h TARGET=hw OPT_LVL=3 \
                        # CLK_FRQ=$FREQZ PE_ENTROP=$pe TRGT_PLATFORM=alveo_u200;
                        make hw_gen ncm=$ncm PE=$p CORE_NR=$cn HT=$h FREQ_MHZ=$FREQZ PE_ENTROP=$pe \
                        TRGT_PLATFORM=zcu104;
                done; 
            done;
        done;
    done
done;
