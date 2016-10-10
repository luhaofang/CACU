/*
 Copyright (c) 2016, David lu
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of the <organization> nor the
 names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

namespace mycnn {

#if GPU_MODE

//caculate sign(I)
extern "C" void BIT_CACU_SIGN_GPU(float_t **&data,unsigned int **&out_data,
		int num, int length);

//caculate sign(W)
extern "C" void BIT_CACU_SIGN_W_GPU(float_t **&data,unsigned int **&out_data,
		int w_num, int length);

//caculate de sign(I)
extern "C" void BIT_CACU_DESIGN_GPU(float_t **&data, float_t **&out_data,
		int num, int length);

//caculate the a*ks_*(modif - 2 * bitcount(k_^x_))
extern "C" void BIT_CACU_COUNT_CONV_GPU(unsigned int **&data,
		unsigned int **&kernels, float_t **&ks, float_t **&a,int num, int motif,
		int w_num, int out_length, int block_size,
		float_t **&out_data);

#else

//caculate sign(I)
void BIT_CACU_SIGN(vector<vec_t> &data, vector<dynamic_bitset<>> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() == out_data[0].size());

	int dim = data[0].size();

	float_t *sdp;

	for (int num = 0; num < data.size(); num++) {
		sdp = &data[num][0];
		//iteration for feature map
		for (int f = 0; f < dim; f++) {
			if (*(sdp + f) > 0)
				out_data[num].flip(f);
		}
	}
}

//caculate sign(W)
void BIT_CACU_SIGN_W(vector<vec_t> &data, bin_param &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	if (data[0].size() % BIN_SIZE == 0)
		assert(data[0].size() / BIN_SIZE == out_data[0].size());
	else
		assert(data[0].size() / BIN_SIZE + 1 == out_data[0].size());

	int dim = data[0].size();

	float_t *sdp;
	unsigned int *snp;

	int flag = BIN_SIZE - 1;

	int end = dim - 1;

	int index;

	for (int num = 0; num < data.size(); num++) {
		sdp = &data[num][0];
		snp = &out_data[num][0];
		dynamic_bitset<> sign_(BIN_SIZE);
		//iteration for feature map
		for (int f = 0; f < dim; f++) {
			index = (f % BIN_SIZE);
			if (*(sdp + f) > 0)
				sign_.flip(index);
			if (f % BIN_SIZE == flag || f == end) {
				*snp = (sign_.to_ulong());
				snp += 1;
				sign_.reset();
			}
		}
	}
}

//caculate de sign(I)
void BIT_CACU_DESIGN(vector<vec_t> &data, vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() == out_data[0].size());

	int dim = data[0].size();

	activation::tan_h h;
	float_t *sdp, *snp;

	for (int num = 0; num < data.size(); num++) {
		sdp = &data[num][0];
		snp = &out_data[num][0];
		//iteration for feature map
		for (int f = 0; f < dim; f++) {

			if (abs(*(sdp + f)) < 1.0)
				*(snp + f) = 1.0; //0.2*h.df(*(sdp + f));
			else
				*(snp + f) = 0.0;

		}
	}
}

//caculate the a*ks_*(modif - 2 * bitcount(k_^x_))
void BIT_CACU_COUNT_CONV_CPU(vec_i &data, vector<vec_i> &kernels, vec_t &ks,
		vector<vec_t> &a, int motif, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0 && kernels.size() > 0);
	assert(data.size() / kernels[0].size() * a.size() == out_data.size());
	assert(kernels.size() == a.size());
	assert(data.size() % kernels[0].size() == 0);

	unsigned int *sp;
	float_t *snp, *skp;
	sp = &data[0];
	snp = &out_data[0];
	skp = &ks[0];
	float_t sum = 0;
	int block_size = kernels[0].size();
	int end = data.size() - 1;
	int flag = block_size - 1;
	int kernel_index, ks_index;
	int output_channel = kernels.size();
	//bitcount
	for (int i = 0; i < data.size(); i++) {
		kernel_index = i % block_size;
		for (int j = 0; j < output_channel; j++) {

			*(snp + j) += float_t(
					bitcount(*(sp + i) ^ (*(&kernels[j][0] + kernel_index))));
		}
		if (kernel_index == flag)
			snp += output_channel;
	}
	snp = &out_data[0];
	for (int i = 0; i < out_data.size(); i++) {
		ks_index = i / output_channel;
		kernel_index = i % output_channel;
		//printf("%d,%d\n", kernel_index, ks_index);
		//printf("%f,%f,%f\n", (motif - 2 * (*(snp + i))), a[kernel_index][0], *(skp + ks_index));
		*(snp + i) = a[kernel_index][0] * (*(skp + ks_index))
				* (motif - 2 * (*(snp + i)));
		//printf("%f,%f,%f\n", *(snp + i), a[kernel_index][0], *(skp + ks_index));
	}
}

#endif

}
;
