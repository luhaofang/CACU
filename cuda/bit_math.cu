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

#include <stdio.h>
#include <stdlib.h>

#define BIN_SIZE 32

using namespace std;

#define CHECK(res) if(res!=cudaSuccess){exit(-1);}

#define BLOCKNUM 1024
#define THREADNUM 128

__global__ void _k_BIT_CACU_SIGN_GPU(float_t **data_input,
		unsigned int **data_output, int num, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int out_start, in_start;

	int data_row;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		out_start = j % length;
		in_start = j % length;
		data_row = j / length;
		if (data_input[data_row][in_start] > 0)
			data_output[data_row][out_start] = 1;
	}
}

//caculate sign(I)
extern "C" void BIT_CACU_SIGN_GPU(float_t **&data, unsigned int **&out_data,
		int num, int length) {

	_k_BIT_CACU_SIGN_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data, num,
			length);

	cudaThreadSynchronize();

}

__global__ void _k_BIT_CACU_SIGN_W_GPU(float_t *data_input,
		unsigned int *data_output, int num, int length, int out_length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	unsigned int sp[BIN_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned int R[BIN_SIZE] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
			2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
			1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864,
			134217728, 268435456, 536870912, 1073741824, 2147483648 };

	int end = length - 1;

	int in_start, data;
	int data_row;
	for (int j = threadid; j < num * out_length; j += BLOCKNUM * THREADNUM) {
		data_row = j / out_length;
		in_start = j % out_length * BIN_SIZE;
		data = 0;
		for (int i = 0; i < BIN_SIZE; i++) {
			if (data_input[data_row*length + in_start + i] > 0)
				sp[i] = 1;
			else
				sp[i] = 0;
			if (in_start + i == end)
				break;
		}
		for (int i = 0; i < BIN_SIZE; i++) {
			data += R[i] * sp[i];
		}
		data_output[j] = data;
		for (int m = 0; m < BIN_SIZE; m++)
			sp[m] = 0;
	}

	//cudaFree(sp);
	//cudaFree(R);

}

//caculate sign(W)
extern "C" void BIT_CACU_SIGN_W_GPU(float_t *&data, unsigned int *&out_data,
		int w_num, int length) {

	int out_length = length / BIN_SIZE;

	_k_BIT_CACU_SIGN_W_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data, w_num,
			length, out_length);

	cudaThreadSynchronize();

}

__global__ void _k_BIT_CACU_DESIGN_GPU(float_t **data_input,
		float_t **data_output, int num, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int out_start, in_start;
	int data_row;
	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_row = j / length;
		out_start = j % length;
		in_start = j % length;
		if (abs(data_input[data_row][in_start]) < 1)
			data_output[data_row][out_start] = 1.0;
		else
			data_output[data_row][out_start] = 0.0;
	}
}

//caculate de sign(I)
extern "C" void BIT_CACU_DESIGN_GPU(float_t **&data, float_t **&out_data,
		int num, int length) {

	_k_BIT_CACU_DESIGN_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data, num,
			length);

	cudaThreadSynchronize();
}

__global__ void _k_BIT_CACU_COUNT_CONV_GPU(unsigned int *data_input,
		float_t *data_output, unsigned int *kernels, float_t *ks, float_t *a,
		int num, int kernels_num, int out_length, int motif, int block_size) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int in_start, bit_count, index, xnor, sum;
	int data_row, data_col;

	int ks_length = out_length / kernels_num;
	int indata_length = ks_length * block_size;
	int kernel_length = block_size;

	for (int j = threadid; j < out_length * num; j += BLOCKNUM * THREADNUM) {
		data_row = j / out_length;
		data_col = j % out_length;
		in_start = data_col / kernels_num * block_size;
		index = data_col % kernels_num;
		sum = 0;
		for (int i = 0; i < block_size; i++) {
			xnor = data_input[data_row * indata_length + in_start + i]
					^ kernels[index * kernel_length + i];

			for (bit_count = 0; xnor != 0; xnor &= (xnor - 1))
				bit_count++;
			sum += bit_count;
		}
		data_output[j] = (float_t) (motif - 2 * sum)
				* ks[data_row * ks_length + data_col / kernels_num] * a[index];
	}
}

//caculate the a*ks_*(modif - 2 * bitcount(k_^x_))
//block_size is the kernel_size*kernel_size*channel / BIN_SIZE
extern "C" void BIT_CACU_COUNT_CONV_GPU(unsigned int *&data,
		unsigned int *&kernels, float_t *&ks, float_t *&a, int num, int motif,
		int kernels_num, int out_length, int block_size, float_t *&out_data) {

	_k_BIT_CACU_COUNT_CONV_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data,
			kernels, ks, a,num, kernels_num, out_length, block_size, motif);

	cudaThreadSynchronize();

}

