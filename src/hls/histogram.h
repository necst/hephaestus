/******************************************
*MIT License
*
# *Copyright (c) [2023] [Giuseppe Sorrentino, Marco Venere, Davide Conficconi, Eleonora D'Arnese, Marco Domenico Santambrogio]
# *Copyright (c) [2022] [Davide Conficconi, Eleonora D'Arnese, Emanuele Del Sozzo, Donatella Sciuto, Marco D. Santambrogio]

*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/
/***************************************************************
*
* High-Level-Synthesis header file for Mutual Information computation
*
****************************************************************/
#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "mutual_info.hpp"
#include "hls_stream.h"



template<typename Tin, unsigned int dim, unsigned int slice, typename Thist, typename Tout, unsigned int bitsThist>
void joint_histogram(hls::stream<Tin> &ref_stream, hls::stream<Tin> &flt_stream, hls::stream<Tout> &j_h_stream, bool end_reset){
	//The end_reset params resets the content of j_h;
	//If not set, the PE memories will accumulate over different iterations.
	//It is set to 1 at the end of the data flow.
	static Thist j_h[HIST_PE][J_HISTO_ROWS][J_HISTO_COLS] = {0};
#pragma HLS ARRAY_PARTITION variable=j_h complete dim=1
#pragma HLS ARRAY_PARTITION variable=j_h cyclic factor=ENTROPY_PE_CONST dim=3

	Tin old_x = 0, old_y = 0;
	Thist acc = 0;

#pragma HLS DEPENDENCE variable=j_h intra RAW false
	HIST:for(int i = 0; i < dim; i++){
#pragma HLS PIPELINE II=1
		Tin curr_x = ref_stream.read();
		Tin curr_y = flt_stream.read();

		if(curr_x == old_x && curr_y == old_y){
			acc += 1;
		} else {
			j_h[slice][old_x][old_y] = acc;
			acc = j_h[slice][curr_x][curr_y] + 1;
		}
		old_x = curr_x;
		old_y = curr_y;
	}

	j_h[slice][old_x][old_y] = acc;
	/*if(end_reset)
		memset(j_h[slice], 0, sizeof(j_h[slice]));*/
	WRITE_OUT:for(int i = 0; i < J_HISTO_ROWS; i++){
		for(int j = 0; j < J_HISTO_COLS; j+=ENTROPY_PE){
#pragma HLS PIPELINE
			Tout val = 0;
			for(int k = 0; k < ENTROPY_PE; k++){
				val.range((k+1)*bitsThist-1, k*bitsThist) = j_h[slice][i][j + k];
				if(end_reset)
					j_h[slice][i][j + k] = 0;
			}

			j_h_stream.write(val);
		}
	}

}


template<typename Tin, unsigned int dim, typename Tout, unsigned int STREAM, typename TtmpIn, unsigned int bitsTtmpIn, typename TtmpOut, unsigned int bitsTtmpOut>
void sum_joint_histogram(hls::stream<Tin> in_stream[STREAM], hls::stream<Tout> &j_h_stream){

	static TtmpOut tmp[ENTROPY_PE];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

	for(int i = 0; i < dim; i++){
#pragma HLS PIPELINE
		Tout out = 0;
		for(int j = 0; j < STREAM; j++){
			Tin elem = in_stream[j].read();
			for(int k = 0; k < ENTROPY_PE; k++){
				TtmpIn unpacked = elem.range((k+1)*bitsTtmpIn-1, k*bitsTtmpIn);
				tmp[k] += unpacked;
			}
		}
		for(int k = 0; k < ENTROPY_PE; k++){
			out.range((k+1)*bitsTtmpOut-1, k*bitsTtmpOut) = tmp[k];
			tmp[k] = 0;
		}
		j_h_stream.write(out);
	}
}


template<typename Tin, unsigned int dim, typename Tout, typename TtmpIn, unsigned int bitsTtmpIn, typename TtmpOut, unsigned int bitsTtmpOut>
void convert(hls::stream<Tin> &in_stream, hls::stream<Tout> &out_stream){
	for(int i = 0; i < dim; i++){
#pragma HLS PIPELINE
		Tin in = in_stream.read();
		Tout out = 0;
		for(int k = 0; k < ENTROPY_PE; k++){
			TtmpIn unpackedIn = in.range((k+1)*bitsTtmpIn-1, k*bitsTtmpIn);
			unsigned int tmp = unpackedIn;
			out.range((k+1)*bitsTtmpOut-1, k*bitsTtmpOut) = tmp;
		}
		out_stream.write(out);
	}

}


template<typename Tin, unsigned int dim, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_1(hls::stream<Tin> ref_pe_stream[1], hls::stream<Tin> flt_pe_stream[1], hls::stream<Tout> j_h_pe_stream[1], bool end_reset){
#pragma HLS INLINE

	joint_histogram<Tin, dim, 0, Thist, Tout, bitsThist>(ref_pe_stream[0], flt_pe_stream[0], j_h_pe_stream[0], end_reset);

}


template<typename Tin, unsigned int dim, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_2(hls::stream<Tin> ref_pe_stream[2], hls::stream<Tin> flt_pe_stream[2], hls::stream<Tout> j_h_pe_stream[2], bool end_reset){
#pragma HLS INLINE

	joint_histogram<Tin, dim, 0, Thist, Tout, bitsThist>(ref_pe_stream[0], flt_pe_stream[0], j_h_pe_stream[0], end_reset);
	joint_histogram<Tin, dim, 1, Thist, Tout, bitsThist>(ref_pe_stream[1], flt_pe_stream[1], j_h_pe_stream[1], end_reset);

}


template<typename Tin, unsigned int dim, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_4(hls::stream<Tin> ref_pe_stream[4], hls::stream<Tin> flt_pe_stream[4], hls::stream<Tout> j_h_pe_stream[4], bool end_reset){
#pragma HLS INLINE

	joint_histogram<Tin, dim, 0, Thist, Tout, bitsThist>(ref_pe_stream[0], flt_pe_stream[0], j_h_pe_stream[0], end_reset);
	joint_histogram<Tin, dim, 1, Thist, Tout, bitsThist>(ref_pe_stream[1], flt_pe_stream[1], j_h_pe_stream[1], end_reset);
	joint_histogram<Tin, dim, 2, Thist, Tout, bitsThist>(ref_pe_stream[2], flt_pe_stream[2], j_h_pe_stream[2], end_reset);
	joint_histogram<Tin, dim, 3, Thist, Tout, bitsThist>(ref_pe_stream[3], flt_pe_stream[3], j_h_pe_stream[3], end_reset);

}


template<typename Tin, unsigned int dim, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_8(hls::stream<Tin> ref_pe_stream[8], hls::stream<Tin> flt_pe_stream[8], hls::stream<Tout> j_h_pe_stream[8], bool end_reset){
#pragma HLS INLINE

	joint_histogram<Tin, dim, 0, Thist, Tout, bitsThist>(ref_pe_stream[0], flt_pe_stream[0], j_h_pe_stream[0], end_reset);
	joint_histogram<Tin, dim, 1, Thist, Tout, bitsThist>(ref_pe_stream[1], flt_pe_stream[1], j_h_pe_stream[1], end_reset);
	joint_histogram<Tin, dim, 2, Thist, Tout, bitsThist>(ref_pe_stream[2], flt_pe_stream[2], j_h_pe_stream[2], end_reset);
	joint_histogram<Tin, dim, 3, Thist, Tout, bitsThist>(ref_pe_stream[3], flt_pe_stream[3], j_h_pe_stream[3], end_reset);
	joint_histogram<Tin, dim, 4, Thist, Tout, bitsThist>(ref_pe_stream[4], flt_pe_stream[4], j_h_pe_stream[4], end_reset);
	joint_histogram<Tin, dim, 5, Thist, Tout, bitsThist>(ref_pe_stream[5], flt_pe_stream[5], j_h_pe_stream[5], end_reset);
	joint_histogram<Tin, dim, 6, Thist, Tout, bitsThist>(ref_pe_stream[6], flt_pe_stream[6], j_h_pe_stream[6], end_reset);
	joint_histogram<Tin, dim, 7, Thist, Tout, bitsThist>(ref_pe_stream[7], flt_pe_stream[7], j_h_pe_stream[7], end_reset);

}


template<typename Tin, unsigned int dim, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_16(hls::stream<Tin> ref_pe_stream[16], hls::stream<Tin> flt_pe_stream[16], hls::stream<Tout> j_h_pe_stream[16], bool end_reset){
#pragma HLS INLINE

	joint_histogram<Tin, dim, 0, Thist, Tout, bitsThist>(ref_pe_stream[0], flt_pe_stream[0], j_h_pe_stream[0], end_reset);
	joint_histogram<Tin, dim, 1, Thist, Tout, bitsThist>(ref_pe_stream[1], flt_pe_stream[1], j_h_pe_stream[1], end_reset);
	joint_histogram<Tin, dim, 2, Thist, Tout, bitsThist>(ref_pe_stream[2], flt_pe_stream[2], j_h_pe_stream[2], end_reset);
	joint_histogram<Tin, dim, 3, Thist, Tout, bitsThist>(ref_pe_stream[3], flt_pe_stream[3], j_h_pe_stream[3], end_reset);
	joint_histogram<Tin, dim, 4, Thist, Tout, bitsThist>(ref_pe_stream[4], flt_pe_stream[4], j_h_pe_stream[4], end_reset);
	joint_histogram<Tin, dim, 5, Thist, Tout, bitsThist>(ref_pe_stream[5], flt_pe_stream[5], j_h_pe_stream[5], end_reset);
	joint_histogram<Tin, dim, 6, Thist, Tout, bitsThist>(ref_pe_stream[6], flt_pe_stream[6], j_h_pe_stream[6], end_reset);
	joint_histogram<Tin, dim, 7, Thist, Tout, bitsThist>(ref_pe_stream[7], flt_pe_stream[7], j_h_pe_stream[7], end_reset);
	joint_histogram<Tin, dim, 8, Thist, Tout, bitsThist>(ref_pe_stream[8], flt_pe_stream[8], j_h_pe_stream[8], end_reset);
	joint_histogram<Tin, dim, 9, Thist, Tout, bitsThist>(ref_pe_stream[9], flt_pe_stream[9], j_h_pe_stream[9], end_reset);
	joint_histogram<Tin, dim, 10, Thist, Tout, bitsThist>(ref_pe_stream[10], flt_pe_stream[10], j_h_pe_stream[10], end_reset);
	joint_histogram<Tin, dim, 11, Thist, Tout, bitsThist>(ref_pe_stream[11], flt_pe_stream[11], j_h_pe_stream[11], end_reset);
	joint_histogram<Tin, dim, 12, Thist, Tout, bitsThist>(ref_pe_stream[12], flt_pe_stream[12], j_h_pe_stream[12], end_reset);
	joint_histogram<Tin, dim, 13, Thist, Tout, bitsThist>(ref_pe_stream[13], flt_pe_stream[13], j_h_pe_stream[13], end_reset);
	joint_histogram<Tin, dim, 14, Thist, Tout, bitsThist>(ref_pe_stream[14], flt_pe_stream[14], j_h_pe_stream[14], end_reset);
	joint_histogram<Tin, dim, 15, Thist, Tout, bitsThist>(ref_pe_stream[15], flt_pe_stream[15], j_h_pe_stream[15], end_reset);

}


template<typename Tin, unsigned int dim, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_32(hls::stream<Tin> ref_pe_stream[32], hls::stream<Tin> flt_pe_stream[32], hls::stream<Tout> j_h_pe_stream[32], bool end_reset){
#pragma HLS INLINE


	joint_histogram<Tin, dim, 0, Thist, Tout, bitsThist>(ref_pe_stream[0], flt_pe_stream[0], j_h_pe_stream[0], end_reset);
	joint_histogram<Tin, dim, 1, Thist, Tout, bitsThist>(ref_pe_stream[1], flt_pe_stream[1], j_h_pe_stream[1], end_reset);
	joint_histogram<Tin, dim, 2, Thist, Tout, bitsThist>(ref_pe_stream[2], flt_pe_stream[2], j_h_pe_stream[2], end_reset);
	joint_histogram<Tin, dim, 3, Thist, Tout, bitsThist>(ref_pe_stream[3], flt_pe_stream[3], j_h_pe_stream[3], end_reset);
	joint_histogram<Tin, dim, 4, Thist, Tout, bitsThist>(ref_pe_stream[4], flt_pe_stream[4], j_h_pe_stream[4], end_reset);
	joint_histogram<Tin, dim, 5, Thist, Tout, bitsThist>(ref_pe_stream[5], flt_pe_stream[5], j_h_pe_stream[5], end_reset);
	joint_histogram<Tin, dim, 6, Thist, Tout, bitsThist>(ref_pe_stream[6], flt_pe_stream[6], j_h_pe_stream[6], end_reset);
	joint_histogram<Tin, dim, 7, Thist, Tout, bitsThist>(ref_pe_stream[7], flt_pe_stream[7], j_h_pe_stream[7], end_reset);
	joint_histogram<Tin, dim, 8, Thist, Tout, bitsThist>(ref_pe_stream[8], flt_pe_stream[8], j_h_pe_stream[8], end_reset);
	joint_histogram<Tin, dim, 9, Thist, Tout, bitsThist>(ref_pe_stream[9], flt_pe_stream[9], j_h_pe_stream[9], end_reset);
	joint_histogram<Tin, dim, 10, Thist, Tout, bitsThist>(ref_pe_stream[10], flt_pe_stream[10], j_h_pe_stream[10], end_reset);
	joint_histogram<Tin, dim, 11, Thist, Tout, bitsThist>(ref_pe_stream[11], flt_pe_stream[11], j_h_pe_stream[11], end_reset);
	joint_histogram<Tin, dim, 12, Thist, Tout, bitsThist>(ref_pe_stream[12], flt_pe_stream[12], j_h_pe_stream[12], end_reset);
	joint_histogram<Tin, dim, 13, Thist, Tout, bitsThist>(ref_pe_stream[13], flt_pe_stream[13], j_h_pe_stream[13], end_reset);
	joint_histogram<Tin, dim, 14, Thist, Tout, bitsThist>(ref_pe_stream[14], flt_pe_stream[14], j_h_pe_stream[14], end_reset);
	joint_histogram<Tin, dim, 15, Thist, Tout, bitsThist>(ref_pe_stream[15], flt_pe_stream[15], j_h_pe_stream[15], end_reset);
	joint_histogram<Tin, dim, 16, Thist, Tout, bitsThist>(ref_pe_stream[16], flt_pe_stream[16], j_h_pe_stream[16], end_reset);
	joint_histogram<Tin, dim, 17, Thist, Tout, bitsThist>(ref_pe_stream[17], flt_pe_stream[17], j_h_pe_stream[17], end_reset);
	joint_histogram<Tin, dim, 18, Thist, Tout, bitsThist>(ref_pe_stream[18], flt_pe_stream[18], j_h_pe_stream[18], end_reset);
	joint_histogram<Tin, dim, 19, Thist, Tout, bitsThist>(ref_pe_stream[19], flt_pe_stream[19], j_h_pe_stream[19], end_reset);
	joint_histogram<Tin, dim, 20, Thist, Tout, bitsThist>(ref_pe_stream[20], flt_pe_stream[20], j_h_pe_stream[20], end_reset);
	joint_histogram<Tin, dim, 21, Thist, Tout, bitsThist>(ref_pe_stream[21], flt_pe_stream[21], j_h_pe_stream[21], end_reset);
	joint_histogram<Tin, dim, 22, Thist, Tout, bitsThist>(ref_pe_stream[22], flt_pe_stream[22], j_h_pe_stream[22], end_reset);
	joint_histogram<Tin, dim, 23, Thist, Tout, bitsThist>(ref_pe_stream[23], flt_pe_stream[23], j_h_pe_stream[23], end_reset);
	joint_histogram<Tin, dim, 24, Thist, Tout, bitsThist>(ref_pe_stream[24], flt_pe_stream[24], j_h_pe_stream[24], end_reset);
	joint_histogram<Tin, dim, 25, Thist, Tout, bitsThist>(ref_pe_stream[25], flt_pe_stream[25], j_h_pe_stream[25], end_reset);
	joint_histogram<Tin, dim, 26, Thist, Tout, bitsThist>(ref_pe_stream[26], flt_pe_stream[26], j_h_pe_stream[26], end_reset);
	joint_histogram<Tin, dim, 27, Thist, Tout, bitsThist>(ref_pe_stream[27], flt_pe_stream[27], j_h_pe_stream[27], end_reset);
	joint_histogram<Tin, dim, 28, Thist, Tout, bitsThist>(ref_pe_stream[28], flt_pe_stream[28], j_h_pe_stream[28], end_reset);
	joint_histogram<Tin, dim, 29, Thist, Tout, bitsThist>(ref_pe_stream[29], flt_pe_stream[29], j_h_pe_stream[29], end_reset);
	joint_histogram<Tin, dim, 30, Thist, Tout, bitsThist>(ref_pe_stream[30], flt_pe_stream[30], j_h_pe_stream[30], end_reset);
	joint_histogram<Tin, dim, 31, Thist, Tout, bitsThist>(ref_pe_stream[31], flt_pe_stream[31], j_h_pe_stream[31], end_reset);

}

template<typename Tin, unsigned int dim, typename Thist, typename Tout, unsigned int bitsThist>
void wrapper_joint_histogram_64(hls::stream<Tin> ref_pe_stream[64], hls::stream<Tin> flt_pe_stream[64], hls::stream<Tout> j_h_pe_stream[64], bool end_reset){
#pragma HLS INLINE


	joint_histogram<Tin, dim, 0, Thist, Tout, bitsThist>(ref_pe_stream[0], flt_pe_stream[0], j_h_pe_stream[0], end_reset);
	joint_histogram<Tin, dim, 1, Thist, Tout, bitsThist>(ref_pe_stream[1], flt_pe_stream[1], j_h_pe_stream[1], end_reset);
	joint_histogram<Tin, dim, 2, Thist, Tout, bitsThist>(ref_pe_stream[2], flt_pe_stream[2], j_h_pe_stream[2], end_reset);
	joint_histogram<Tin, dim, 3, Thist, Tout, bitsThist>(ref_pe_stream[3], flt_pe_stream[3], j_h_pe_stream[3], end_reset);
	joint_histogram<Tin, dim, 4, Thist, Tout, bitsThist>(ref_pe_stream[4], flt_pe_stream[4], j_h_pe_stream[4], end_reset);
	joint_histogram<Tin, dim, 5, Thist, Tout, bitsThist>(ref_pe_stream[5], flt_pe_stream[5], j_h_pe_stream[5], end_reset);
	joint_histogram<Tin, dim, 6, Thist, Tout, bitsThist>(ref_pe_stream[6], flt_pe_stream[6], j_h_pe_stream[6], end_reset);
	joint_histogram<Tin, dim, 7, Thist, Tout, bitsThist>(ref_pe_stream[7], flt_pe_stream[7], j_h_pe_stream[7], end_reset);
	joint_histogram<Tin, dim, 8, Thist, Tout, bitsThist>(ref_pe_stream[8], flt_pe_stream[8], j_h_pe_stream[8], end_reset);
	joint_histogram<Tin, dim, 9, Thist, Tout, bitsThist>(ref_pe_stream[9], flt_pe_stream[9], j_h_pe_stream[9], end_reset);
	joint_histogram<Tin, dim, 10, Thist, Tout, bitsThist>(ref_pe_stream[10], flt_pe_stream[10], j_h_pe_stream[10], end_reset);
	joint_histogram<Tin, dim, 11, Thist, Tout, bitsThist>(ref_pe_stream[11], flt_pe_stream[11], j_h_pe_stream[11], end_reset);
	joint_histogram<Tin, dim, 12, Thist, Tout, bitsThist>(ref_pe_stream[12], flt_pe_stream[12], j_h_pe_stream[12], end_reset);
	joint_histogram<Tin, dim, 13, Thist, Tout, bitsThist>(ref_pe_stream[13], flt_pe_stream[13], j_h_pe_stream[13], end_reset);
	joint_histogram<Tin, dim, 14, Thist, Tout, bitsThist>(ref_pe_stream[14], flt_pe_stream[14], j_h_pe_stream[14], end_reset);
	joint_histogram<Tin, dim, 15, Thist, Tout, bitsThist>(ref_pe_stream[15], flt_pe_stream[15], j_h_pe_stream[15], end_reset);
	joint_histogram<Tin, dim, 16, Thist, Tout, bitsThist>(ref_pe_stream[16], flt_pe_stream[16], j_h_pe_stream[16], end_reset);
	joint_histogram<Tin, dim, 17, Thist, Tout, bitsThist>(ref_pe_stream[17], flt_pe_stream[17], j_h_pe_stream[17], end_reset);
	joint_histogram<Tin, dim, 18, Thist, Tout, bitsThist>(ref_pe_stream[18], flt_pe_stream[18], j_h_pe_stream[18], end_reset);
	joint_histogram<Tin, dim, 19, Thist, Tout, bitsThist>(ref_pe_stream[19], flt_pe_stream[19], j_h_pe_stream[19], end_reset);
	joint_histogram<Tin, dim, 20, Thist, Tout, bitsThist>(ref_pe_stream[20], flt_pe_stream[20], j_h_pe_stream[20], end_reset);
	joint_histogram<Tin, dim, 21, Thist, Tout, bitsThist>(ref_pe_stream[21], flt_pe_stream[21], j_h_pe_stream[21], end_reset);
	joint_histogram<Tin, dim, 22, Thist, Tout, bitsThist>(ref_pe_stream[22], flt_pe_stream[22], j_h_pe_stream[22], end_reset);
	joint_histogram<Tin, dim, 23, Thist, Tout, bitsThist>(ref_pe_stream[23], flt_pe_stream[23], j_h_pe_stream[23], end_reset);
	joint_histogram<Tin, dim, 24, Thist, Tout, bitsThist>(ref_pe_stream[24], flt_pe_stream[24], j_h_pe_stream[24], end_reset);
	joint_histogram<Tin, dim, 25, Thist, Tout, bitsThist>(ref_pe_stream[25], flt_pe_stream[25], j_h_pe_stream[25], end_reset);
	joint_histogram<Tin, dim, 26, Thist, Tout, bitsThist>(ref_pe_stream[26], flt_pe_stream[26], j_h_pe_stream[26], end_reset);
	joint_histogram<Tin, dim, 27, Thist, Tout, bitsThist>(ref_pe_stream[27], flt_pe_stream[27], j_h_pe_stream[27], end_reset);
	joint_histogram<Tin, dim, 28, Thist, Tout, bitsThist>(ref_pe_stream[28], flt_pe_stream[28], j_h_pe_stream[28], end_reset);
	joint_histogram<Tin, dim, 29, Thist, Tout, bitsThist>(ref_pe_stream[29], flt_pe_stream[29], j_h_pe_stream[29], end_reset);
	joint_histogram<Tin, dim, 30, Thist, Tout, bitsThist>(ref_pe_stream[30], flt_pe_stream[30], j_h_pe_stream[30], end_reset);
	joint_histogram<Tin, dim, 31, Thist, Tout, bitsThist>(ref_pe_stream[31], flt_pe_stream[31], j_h_pe_stream[31], end_reset);
	joint_histogram<Tin, dim, 32, Thist, Tout, bitsThist>(ref_pe_stream[32], flt_pe_stream[32], j_h_pe_stream[32], end_reset);
	joint_histogram<Tin, dim, 33, Thist, Tout, bitsThist>(ref_pe_stream[33], flt_pe_stream[33], j_h_pe_stream[33], end_reset);
	joint_histogram<Tin, dim, 34, Thist, Tout, bitsThist>(ref_pe_stream[34], flt_pe_stream[34], j_h_pe_stream[34], end_reset);
	joint_histogram<Tin, dim, 35, Thist, Tout, bitsThist>(ref_pe_stream[35], flt_pe_stream[35], j_h_pe_stream[35], end_reset);
	joint_histogram<Tin, dim, 36, Thist, Tout, bitsThist>(ref_pe_stream[36], flt_pe_stream[36], j_h_pe_stream[36], end_reset);
	joint_histogram<Tin, dim, 37, Thist, Tout, bitsThist>(ref_pe_stream[37], flt_pe_stream[37], j_h_pe_stream[37], end_reset);
	joint_histogram<Tin, dim, 38, Thist, Tout, bitsThist>(ref_pe_stream[38], flt_pe_stream[38], j_h_pe_stream[38], end_reset);
	joint_histogram<Tin, dim, 39, Thist, Tout, bitsThist>(ref_pe_stream[39], flt_pe_stream[39], j_h_pe_stream[39], end_reset);
	joint_histogram<Tin, dim, 40, Thist, Tout, bitsThist>(ref_pe_stream[40], flt_pe_stream[40], j_h_pe_stream[40], end_reset);
	joint_histogram<Tin, dim, 41, Thist, Tout, bitsThist>(ref_pe_stream[41], flt_pe_stream[41], j_h_pe_stream[41], end_reset);
	joint_histogram<Tin, dim, 42, Thist, Tout, bitsThist>(ref_pe_stream[42], flt_pe_stream[42], j_h_pe_stream[42], end_reset);
	joint_histogram<Tin, dim, 43, Thist, Tout, bitsThist>(ref_pe_stream[43], flt_pe_stream[43], j_h_pe_stream[43], end_reset);
	joint_histogram<Tin, dim, 44, Thist, Tout, bitsThist>(ref_pe_stream[44], flt_pe_stream[44], j_h_pe_stream[44], end_reset);
	joint_histogram<Tin, dim, 45, Thist, Tout, bitsThist>(ref_pe_stream[45], flt_pe_stream[45], j_h_pe_stream[45], end_reset);
	joint_histogram<Tin, dim, 46, Thist, Tout, bitsThist>(ref_pe_stream[46], flt_pe_stream[46], j_h_pe_stream[46], end_reset);
	joint_histogram<Tin, dim, 47, Thist, Tout, bitsThist>(ref_pe_stream[47], flt_pe_stream[47], j_h_pe_stream[47], end_reset);
	joint_histogram<Tin, dim, 48, Thist, Tout, bitsThist>(ref_pe_stream[48], flt_pe_stream[48], j_h_pe_stream[48], end_reset);
	joint_histogram<Tin, dim, 49, Thist, Tout, bitsThist>(ref_pe_stream[49], flt_pe_stream[49], j_h_pe_stream[49], end_reset);
	joint_histogram<Tin, dim, 50, Thist, Tout, bitsThist>(ref_pe_stream[50], flt_pe_stream[50], j_h_pe_stream[50], end_reset);
	joint_histogram<Tin, dim, 51, Thist, Tout, bitsThist>(ref_pe_stream[51], flt_pe_stream[51], j_h_pe_stream[51], end_reset);
	joint_histogram<Tin, dim, 52, Thist, Tout, bitsThist>(ref_pe_stream[52], flt_pe_stream[52], j_h_pe_stream[52], end_reset);
	joint_histogram<Tin, dim, 53, Thist, Tout, bitsThist>(ref_pe_stream[53], flt_pe_stream[53], j_h_pe_stream[53], end_reset);
	joint_histogram<Tin, dim, 54, Thist, Tout, bitsThist>(ref_pe_stream[54], flt_pe_stream[54], j_h_pe_stream[54], end_reset);
	joint_histogram<Tin, dim, 55, Thist, Tout, bitsThist>(ref_pe_stream[55], flt_pe_stream[55], j_h_pe_stream[55], end_reset);
	joint_histogram<Tin, dim, 56, Thist, Tout, bitsThist>(ref_pe_stream[56], flt_pe_stream[56], j_h_pe_stream[56], end_reset);
	joint_histogram<Tin, dim, 57, Thist, Tout, bitsThist>(ref_pe_stream[57], flt_pe_stream[57], j_h_pe_stream[57], end_reset);
	joint_histogram<Tin, dim, 58, Thist, Tout, bitsThist>(ref_pe_stream[58], flt_pe_stream[58], j_h_pe_stream[58], end_reset);
	joint_histogram<Tin, dim, 59, Thist, Tout, bitsThist>(ref_pe_stream[59], flt_pe_stream[59], j_h_pe_stream[59], end_reset);
	joint_histogram<Tin, dim, 60, Thist, Tout, bitsThist>(ref_pe_stream[60], flt_pe_stream[60], j_h_pe_stream[60], end_reset);
	joint_histogram<Tin, dim, 61, Thist, Tout, bitsThist>(ref_pe_stream[61], flt_pe_stream[61], j_h_pe_stream[61], end_reset);
	joint_histogram<Tin, dim, 62, Thist, Tout, bitsThist>(ref_pe_stream[62], flt_pe_stream[62], j_h_pe_stream[62], end_reset);
	joint_histogram<Tin, dim, 63, Thist, Tout, bitsThist>(ref_pe_stream[63], flt_pe_stream[63], j_h_pe_stream[63], end_reset);

}
#endif // HISTOGRAM_H
