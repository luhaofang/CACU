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
#include <assert.h>

#define BIN_SIZE 32

using namespace std;

#define CHECK(res) if(res!=cudaSuccess){exit(-1);}

#define BLOCKNUM 512
#define THREADNUM 512

__global__ void _k_CACU_SUM_SIZE_GPU(float_t **data, int num, int sum_size,
		int length, int out_length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int start_out, start_in;

	int data_row, data_col;

	for (int i = threadid; i < num * out_length; i += BLOCKNUM * THREADNUM) {
		data_row = i / out_length;
		data_col = i % out_length;
		start_out = data_col;
		start_in = data_col * sum_size;

		out_data[data_row][start_out] = 0.0;

		for (int j = 0; j < sum_size; j++)
			out_data[data_row][start_out] += data[data_row][start_in + j];
	}
}

//vec_t(size) -> vec_t(size/sum_size)
extern "C" void CACU_SUM_SIZE_GPU(float_t **&data, int num, int sum_size,
		int length, int out_length, float_t **&out_data) {

	assert(length / sum_size == out_length);
	assert(length % sum_size == 0);

	_k_CACU_SUM_SIZE_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, sum_size,
			length, out_length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_MEAN_GPU(float_t *data, int num, int length,
		float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t shared_data[];

	for (int i = bid; i < num; i += BLOCKNUM) {

		shared_data[tid] = 0;

		for (int j = tid; j < length; j += THREADNUM) {

			shared_data[tid] += data[i * length + j];
		}

		__syncthreads();

		if (tid == 0) {
			for (int j = 1; j < THREADNUM; j++)
				shared_data[0] += shared_data[j];
			out_data[i] = shared_data[0] / length;
		}
	}
}

//vec_t(size) -> vec_t(size/sum_size)
extern "C" void CACU_MEAN_GPU(float_t *&data, int num, int length,
		float_t *&out_data) {

	_k_CACU_MEAN_GPU<<<BLOCKNUM, THREADNUM, THREADNUM * sizeof(float_t)>>>(data,
			num, length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUM_SIZE_ABS_GPU(float_t *data, int num, int sum_size,
		int length, int out_length, float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int start_in;

	int data_row, data_col;

	for (int i = threadid; i < num * out_length; i += BLOCKNUM * THREADNUM) {
		data_row = i / out_length;
		data_col = i % out_length;
		start_in = data_col * sum_size;

		out_data[i] = 0.0;

		for (int j = 0; j < sum_size; j++)
			out_data[i] += abs(data[data_row * length + start_in + j]);
	}
}

//vec_t(size) -> vec_t(size/sum_size)
extern "C" void CACU_SUM_SIZE_ABS_GPU(float_t *&data, int num, int sum_size,
		int length, int out_length, float_t *&out_data) {

	assert(length / sum_size == out_length);
	assert(length % sum_size == 0);

	_k_CACU_SUM_SIZE_ABS_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, sum_size,
			length, out_length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_MEAN_CHANNEL_GPU(float_t **data, float_t denominator,
		int num, int dim, int channel, float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t share_data[];

	int data_row, data_col;

	share_data[tid] = 0;

	for (int i = tid; i < dim * num; i += THREADNUM)
	{
		data_row = i / dim;
		data_col = i % dim;

		share_data[tid] += data[data_row][data_col * channel + bid];
	}

	__syncthreads();

	if (tid == 0) {
		for (int i = 1; i < THREADNUM; i++) {
			share_data[0] += share_data[i];
		}
		out_data[bid] = share_data[0] / denominator;
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the means for batch_size
extern "C" void CACU_MEAN_CHANNEL_GPU(float_t **&data, int num, int length,
		int channel, float_t *&out_data) {

	assert(length % channel == 0);

	int dim = length / channel;

	float_t denominator = (float_t) dim * num;

	_k_CACU_MEAN_CHANNEL_GPU<<<channel, THREADNUM, THREADNUM * sizeof(float_t)>>>(
			data, denominator, num, dim, channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_VARIANCE_CHANNEL_GPU(float_t **data,
		float_t denominator, int num, int dim, int channel, float_t *mean,
		float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t share_data[];

	int data_row, data_col;

	share_data[tid] = 0;

	for (int i = tid; i < dim * num; i += THREADNUM)
	{
		data_row = i / dim;
		data_col = i % dim;

		share_data[tid] += ((data[data_row][data_col * channel + bid]
				- mean[bid])
				* (data[data_row][data_col * channel + bid] - mean[bid]));
	}

	__syncthreads();

	if (tid == 0) {
		for (int i = 1; i < THREADNUM; i++) {
			share_data[0] += share_data[i];
		}
		out_data[bid] = share_data[0] / denominator;
	}

}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the variance for batch_size
extern "C" void CACU_VARIANCE_CHANNEL_GPU(float_t **&data, float_t *&mean,
		int num, int length, int channel, float_t *&out_data) {

	assert(length % channel == 0);

	int dim = length / channel;

	float_t denominator = (float_t) dim * num;

	_k_CACU_VARIANCE_CHANNEL_GPU<<<channel, THREADNUM,
	THREADNUM * sizeof(float_t)>>>(data, denominator, num, dim, channel,
			mean, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DOT_GPU(float_t **data, float_t **scale, int num,
		int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		out_data[data_row][data_col] = data[data_row][data_col]
				* scale[data_row][data_col];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)-207.705643,1:-539.477417,2:-787.299805,

//caculate the channel's scale for batch_size
extern "C" void CACU_DOT_GPU(float_t **&data, float_t **&scale, int num,
		int length, float_t **&out_data) {

	_k_CACU_DOT_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, scale, num, length,
			out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SQRT_GPU(float_t **data, int num, int length,
		float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		out_data[data_row][data_col] = sqrt(data[data_row][data_col]);
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's scale for batch_size
extern "C" void CACU_SQRT_GPU(float_t **&data, int num, int length,
		float_t **&out_data) {
	_k_CACU_SQRT_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SCALE_GPU(float_t **data, float_t *scale, int num,
		int length, int channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		out_data[data_row][data_col] = data[data_row][data_col]
				* scale[data_col % channel];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's scale for batch_size
extern "C" void CACU_SCALE_GPU(float_t **&data, float_t *&scale, int num,
		int length, int channel, float_t **&out_data) {

	assert(length % channel == 0);

	_k_CACU_SCALE_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, scale, num, length,
			channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SCALE_GPU_D(float_t **data, float_t **scale, int num,
		int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;
	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		out_data[data_row][data_col] = data[data_row][data_col]
				* scale[data_row][data_col];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix A*B
extern "C" void CACU_SCALE_GPU_D(float_t **&data, float_t **&scale, int num,
		int length, float_t **&out_data) {

	_k_CACU_SCALE_GPU_D<<<BLOCKNUM, THREADNUM, 0>>>(data, scale, num, length,
			out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SCALE_GPU_A(float_t **data, float_t scale, int num,
		int length, float_t **out_data, int add) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		if (add == 0)
			out_data[data_row][data_col] = data[data_row][data_col] * scale;
		else
			out_data[data_row][data_col] += data[data_row][data_col] * scale;
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix scale*B
extern "C" void CACU_SCALE_GPU_A(float_t **&data, float_t scale, int num,
		int length, float_t **&out_data, int add) {

	_k_CACU_SCALE_GPU_A<<<BLOCKNUM, THREADNUM, 0>>>(data, scale, num, length,
			out_data, add);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_SCALE_GPU_B(float_t **data, float_t **scale, int num,
		int dim, int channel, float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t share_data[];

	int data_row, data_col;

	share_data[tid] = 0;

	for (int i = tid; i < dim * num; i += THREADNUM)
	{
		data_row = i / dim;
		data_col = i % dim;

		share_data[tid] += (data[data_row][data_col * channel + bid]
				* scale[data_row][data_col * channel + bid]);
	}

	__syncthreads();

	if (tid == 0) {
		for (int i = 1; i < THREADNUM; i++) {
			share_data[0] += share_data[i];
		}
		out_data[bid] = share_data[0];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel' scale_sum  for batch_size
extern "C" void CACU_SCALE_GPU_B(float_t **&data, float_t **&scale, int num,
		int length, int channel, float_t *&out_data) {

	assert(length % channel == 0);

	int dim = length / channel;

	_k_CACU_SCALE_GPU_B<<<channel, THREADNUM, THREADNUM * sizeof(float_t)>>>(
			data, scale, num, dim, channel, out_data);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_SUM_GPU(float_t **data, float_t *bias, int num,
		int length, int channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		out_data[data_row][data_col] = data[data_row][data_col]
				+ bias[data_col % channel];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum bias for batch_size
extern "C" void CACU_SUM_GPU(float_t **&data, float_t *&bias, int num,
		int length, int channel, float_t **&out_data) {

	_k_CACU_SUM_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, bias, num, length,
			channel, out_data);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_SUM_GPU_B(float_t **data, int num, int dim, int channel,
		float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t share_data[];

	int data_row, data_col;

	share_data[tid] = 0;

	for (int i = tid; i < dim * num; i += THREADNUM)
	{
		data_row = i / dim;
		data_col = i % dim;

		share_data[tid] += data[data_row][data_col * channel + bid];
	}

	__syncthreads();

	if (tid == 0) {
		for (int i = 1; i < THREADNUM; i++) {
			share_data[0] += share_data[i];
		}
		out_data[bid] = share_data[0];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum for batch_size
extern "C" void CACU_SUM_GPU_B(float_t **&data, int num, int length,
		int channel, float_t *&out_data) {

	assert(length % channel == 0);

	int dim = length / channel;

	_k_CACU_SUM_GPU_B<<<channel, THREADNUM, THREADNUM * sizeof(float_t)>>>(data,
			num, dim, channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUM_GPU_C(float_t **data, int num, int out_length,
		int channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * out_length; i += BLOCKNUM * THREADNUM) {

		data_row = i / out_length;
		data_col = i % out_length;
		for (int j = 0; j < channel; j++) {
			out_data[data_row][data_col] += data[data_row][data_col * channel
					+ j];
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum for every sample
extern "C" void CACU_SUM_GPU_C(float_t **&data, int num, int length,
		int out_length, int channel, float_t **&out_data) {

	assert(length % channel == 0);
	assert(length / channel == out_length);

	_k_CACU_SUM_GPU_C<<<BLOCKNUM, THREADNUM, 0>>>(data, num, out_length,
			channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUM_GPU_R(float_t **data, float_t **bias, int num,
		int output_channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < output_channel; i += BLOCKNUM * THREADNUM) {

		for (int n = 0; n < num; n++)

			out_data[i][0] = data[i][0] + bias[n][i];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum bias for batch_size
extern "C" void CACU_SUM_GPU_R(float_t **&data, float_t **&bias, int num,
		int output_channel, float_t **&out_data) {

	_k_CACU_SUM_GPU_R<<<BLOCKNUM, THREADNUM, 0>>>(data, bias, num,
			output_channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUM_ABS_GPU(float_t **data, int num, int out_length,
		int channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * out_length; i += BLOCKNUM * THREADNUM) {

		data_row = i / out_length;
		data_col = i % out_length;

		for (int j = 0; j < channel; j++) {
			out_data[data_row][data_col] += abs(
					data[data_row][data_col * channel + j]);
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum(abs(x)) for every sample
extern "C" void CACU_SUM_ABS_GPU(float_t **&data, int num, int length,
		int out_length, int channel, float_t **&out_data) {

	assert(length % channel == 0);
	assert(length / channel == out_length);

	_k_CACU_SUM_ABS_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, out_length,
			channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUM_GPU_D(float_t **data, float_t **bias, int num,
		int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		out_data[data_row][data_col] = data[data_row][data_col]
				+ bias[data_row][data_col];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the dim's sum for every batch_size
extern "C" void CACU_SUM_GPU_D(float_t **&data, float_t **&bias, int num,
		int length, float_t **&out_data) {
	_k_CACU_SUM_GPU_D<<<BLOCKNUM, THREADNUM, 0>>>(data, bias, num, length,
			out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUB_GPU(float_t **data, float_t *bias, int num,
		int length, int channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;

		out_data[data_row][data_col] = data[data_row][data_col]
				- bias[data_col % channel];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the subtraction for batch_size
extern "C" void CACU_SUB_GPU(float_t **&data, float_t *&bias, int num,
		int length, int channel, float_t **&out_data) {

	assert(length % channel == 0);

	_k_CACU_SUB_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, bias, num, length,
			channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUB_GPU_D(float_t *data, float_t *bias, int num,
		int length, float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;

		out_data[i] = data[i] - bias[data_row];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the subtraction for batch_size
extern "C" void CACU_SUB_GPU_D(float_t *&data, float_t *&bias, int num,
		int length, float_t *&out_data) {

	_k_CACU_SUB_GPU_D<<<BLOCKNUM, THREADNUM, 0>>>(data, bias, num, length,
			out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DIVISION_GPU(float_t **data, float_t *scale, int num,
		int length, int channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;
		out_data[data_row][data_col] = data[data_row][data_col]
				/ scale[data_col % channel];
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the division for batch_size
extern "C" void CACU_DIVISION_GPU(float_t **&data, float_t *&scale, int num,
		int length, int channel, float_t **&out_data) {

	assert(length % channel == 0);

	_k_CACU_DIVISION_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, scale, num, length,
			channel, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_ROU_GPU(float_t **data, float_t **dx_ba, float_t *mean,
		float_t *variance, int num, int dim, int channel, float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t share_data[];

	int data_row, data_col;

	share_data[tid] = 0;

	for (int i = tid; i < dim * num; i += THREADNUM)
	{
		data_row = i / dim;
		data_col = i % dim;

		share_data[tid] +=
				(data[data_row][data_col * channel + bid] - mean[bid])
						* dx_ba[data_row][data_col * channel + bid]
						* (-0.5
								/ (variance[bid] * variance[bid] * variance[bid]));
	}

	__syncthreads();

	if (tid == 0) {
		for (int i = 1; i < THREADNUM; i++) {
			share_data[0] += share_data[i];
		}
		out_data[bid] = share_data[0];
	}
}

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
extern "C" void CACU_ROU_GPU(float_t **&data, float_t **&dx_ba, float_t *&mean,
		float_t *&variance, int num, int length, int channel,
		float_t *&out_data) {

	assert(length % channel == 0);

	int dim = length / channel;

	_k_CACU_ROU_GPU<<<channel, THREADNUM, THREADNUM * sizeof(float_t)>>>(data,
			dx_ba, mean, variance, num, dim, channel, out_data);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_MU_GPU(float_t **data, float_t **dx_ba, float_t *mean,
		float_t *variance, float_t *rou, int dim, int channel, int num,
		float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t share_data[];

	int data_row, data_col;

	int m = dim * num;

	share_data[tid] = 0;

	for (int i = tid; i < dim * num; i += THREADNUM)
	{
		data_row = i / dim;
		data_col = i % dim;

		share_data[tid] += ((dx_ba[data_row][data_col * channel + bid]
				/ (-variance[bid]))
				+ ((rou[bid] / m)
						* (-2.0
								* (data[data_row][data_col * channel + bid]
										- mean[bid]))));
	}

	__syncthreads();

	if (tid == 0) {
		for (int i = 1; i < THREADNUM; i++) {
			share_data[0] += share_data[i];
		}
		out_data[bid] = share_data[0];
	}
}

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
extern "C" void CACU_MU_GPU(float_t **&data, float_t **&dx_ba, float_t *&mean,
		float_t *&variance, float_t *&rou, int num, int length, int channel,
		float_t *&out_data) {

	assert(length % channel == 0);

	int dim = length / channel;

	_k_CACU_MU_GPU<<<channel, THREADNUM, THREADNUM * sizeof(float_t)>>>(data,
			dx_ba, mean, variance, rou, dim, channel, num, out_data);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_DX_GPU(float_t **data, float_t **dx_ba, float_t *mean,
		float_t *variance, float_t *rou, float_t *mu, int length, int dim,
		int num, int channel, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int c;

	int m = dim * num;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		c = data_col % channel;
		out_data[data_row][data_col] += ((dx_ba[data_row][data_col]
				/ variance[c])
				+ rou[c] * (2.0 * (data[data_row][data_col] - mean[c]) / m)
				+ (mu[c] / m));
	}
}

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
extern "C" void CACU_DX_GPU(float_t **&data, float_t **&dx_ba, float_t *&mean,
		float_t *&variance, float_t *&rou, float_t *&mu, int num, int length,
		int channel, float_t **&out_data) {

	assert(length % channel == 0);

	int dim = length / channel;

	_k_CACU_DX_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, dx_ba, mean, variance, rou,
			mu, length, dim, num, channel, out_data);

	cudaThreadSynchronize();
}

//__global__ void _k_CACU_SCALE_SUM_ROW_GPU(float_t **data, int num,
//		int kernels_num, int sum_size, int out_length, float_t **kernel,
//		float_t **bias, float_t **out_data) {
//
//	int tid = threadIdx.x;
//	int bid = blockIdx.x;
//
//	int threadid = bid * THREADNUM + tid;
//
//	int start_in, start_out;
//
//	int data_row, data_col;
//
//	int c;
//
//	extern __shared__ float_t shared_data[];
//
//	for (int i = bid; i < num * out_length; i += BLOCKNUM) {
//		data_row = i / out_length;
//		data_col = i % out_length;
//
//		start_in = (data_col / kernels_num) * sum_size;
//
//		c = data_col % kernels_num;
//
//		start_out = data_col;
//
//		for (int j = tid; j < sum_size; j += THREADNUM)
//		{
//			shared_data[tid] = data[data_row][start_in + j] * kernel[c][j];
//		}
//
//		__syncthreads();
//
//		if (tid == 0) {
//			for(int i = 1; i < THREADNUM ; i++)
//				shared_data[0] += shared_data[i];
//			out_data[data_row][start_out] = shared_data[0] + bias[c][0];
//		}
//	}
//}
//
//
////caculate the sum(a*x_0i)
//extern "C" void CACU_SCALE_SUM_ROW_GPU(float_t **&data, int num, int sum_size,
//		int kernels_num, int out_length, float_t **&kernels, float_t **&bias,
//		float_t **&out_data) {
//
//	assert(out_length % kernels_num == 0);
//
//	_k_CACU_SCALE_SUM_ROW_GPU<<<BLOCKNUM, THREADNUM, THREADNUM * sizeof(float_t)>>>(
//			data, num, kernels_num, sum_size, out_length, kernels, bias,
//			out_data);
//
//	cudaThreadSynchronize();
//}

__global__ void _k_CACU_SCALE_SUM_ROW_GPU(float_t *data, int num,
		int kernels_num, int sum_size, int out_length, float_t *kernel,
		float_t *bias, float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int start_in;

	int data_row, data_col;

	int c;

	int indata_length = (out_length / kernels_num) * sum_size;

	for (int i = threadid; i < num * out_length; i += BLOCKNUM * THREADNUM) {
		data_row = i / out_length;
		data_col = i % out_length;

		out_data[i] = 0.0;

		start_in = (data_col / kernels_num) * sum_size;

		c = data_col % kernels_num;

		for (int j = 0; j < sum_size; j++) {
			out_data[i] += data[data_row * indata_length + start_in + j]
					* kernel[c * sum_size + j];
		}
		out_data[i] += bias[c];
	}
}

//caculate the sum(a*x_0i)
extern "C" void CACU_SCALE_SUM_ROW_GPU(float_t *&data, int num, int sum_size,
		int kernels_num, int out_length, float_t *&kernels, float_t *&bias,
		float_t *&out_data) {

	assert(out_length % kernels_num == 0);

	_k_CACU_SCALE_SUM_ROW_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num,
			kernels_num, sum_size, out_length, kernels, bias, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DECONV_W_BIN_GPU(float_t *data, float_t *top_diff,
		float_t *a, int num, int kernel_length, int output_dim, int kernels_num,
		float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int dim = output_dim * output_dim;

	int data_row, data_col;

	int data_length = output_dim * output_dim * kernel_length;
	int diff_length = dim * kernels_num;

	float_t crop = 1.0;

	for (int i = threadid; i < kernels_num * kernel_length;
			i += BLOCKNUM * THREADNUM) {

		data_row = i / kernel_length;
		data_col = i % kernel_length;

		out_data[i] = 0.0;

		for (int n = 0; n < num; n++)
			for (int j = 0; j < dim; j++) {
				out_data[i] +=
						data[n * data_length + j * kernel_length + data_col]
								* top_diff[n * diff_length + j * kernels_num
										+ data_row];
			}
		if (abs(out_data[i]) > 1)
			crop = 0.0;
		out_data[i] *= (((float_t) (1.0 / kernel_length) + a[data_row] * crop)
				* ((float_t) kernel_length - (float_t) (1.0)));
	}
}

//caculate the grad_convolution for W
//data : bottom
//top_diff : diffs
//out_data : diff_ws
extern "C" void CACU_DECONV_W_BIN_GPU(float_t *&data, float_t *&top_diff,
		float_t *a, int num, int kernel_size, int kernels_num, int output_dim,
		int channel, int stride, float_t *&out_data) {

	_k_CACU_DECONV_W_BIN_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, top_diff, a, num,
			kernel_size * kernel_size * channel, output_dim, kernels_num,
			out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DECONV_W_B_GPU(float_t *data, float_t *top_diff,
		int num, int kernel_length, int output_dim, int kernels_num,
		float_t *out_data, float_t *bias) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int dim = output_dim * output_dim;

	int data_row, data_col;

	int data_length = output_dim * output_dim * kernel_length;
	int diff_length = dim * kernels_num;

	for (int i = threadid; i < kernels_num * kernel_length;
			i += BLOCKNUM * THREADNUM) {

		data_row = i / kernel_length;
		data_col = i % kernel_length;

		out_data[i] = 0.0;

		for (int n = 0; n < num; n++)
			for (int j = 0; j < dim; j++) {
				out_data[i] +=
						data[n * data_length + j * kernel_length + data_col]
								* top_diff[n * diff_length + j * kernels_num
										+ data_row];
			}
	}

	for (int i = threadid; i < kernels_num; i += BLOCKNUM * THREADNUM) {

		bias[i] = 0.0;

		for (int n = 0; n < num; n++)
			for (int j = 0; j < dim; j++) {
				bias[i] = bias[i]
						+ top_diff[n * diff_length + j * kernels_num + i];
			}
	}
}

//caculate the grad_convolution for W
//data : bottom
//top_diff : diffs
//out_data : diff_ws
extern "C" void CACU_DECONV_W_B_GPU(float_t *&data, float_t *&top_diff, int num,
		int kernel_size, int kernels_num, int output_dim, int channel,
		int stride, float_t *&out_data, float_t *&bias) {

	_k_CACU_DECONV_W_B_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, top_diff, num,
			kernel_size * kernel_size * channel, output_dim, kernels_num,
			out_data, bias);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DECONV_DIFF_GPU(float_t **data, float_t **kernel,
		int num, int channel, int kernels_num, int input_dim, int output_dim,
		int stride, int kernel_size, int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

//the set in the input feature map
	int startset_i, startset_j;
//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
//the count for stride in feature map
	int count_i, count_j;

	int data_row, data_col;

	int k_index, diff_index;

	int c;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		out_data[data_row][data_col] = 0.0;

		startset_i = data_col / (channel * input_dim);
		startset_j = (data_col / channel) % input_dim;
		c = data_col % channel;
		outset_si = startset_i / stride;
		outset_sj = startset_j / stride;

		if (outset_si >= output_dim)
			outset_si = output_dim - 1;
		if (outset_sj >= output_dim)
			outset_sj = output_dim - 1;

		count_i = 0;
		count_j = 0;

		while (outset_si - (count_i + 1) >= 0
				&& ((outset_si - (count_i + 1)) * stride) + kernel_size
						>= startset_i + 1) {
			count_i++;
		}
		while (outset_sj - (count_j + 1) >= 0
				&& ((outset_sj - (count_j + 1)) * stride) + kernel_size
						>= startset_j + 1) {
			count_j++;
		}

		//stride
		for (int mi = 0; mi <= count_i; mi++)
			for (int mj = 0; mj <= count_j; mj++) {
				outset_i = outset_si - mi;
				outset_j = outset_sj - mj;

				k_index = ((startset_i - outset_i * stride) * kernel_size
						+ (startset_j - outset_j * stride)) * channel + c;
				diff_index = (outset_i * output_dim + outset_j) * kernels_num;

				for (int kn = 0; kn < kernels_num; kn++) {
					out_data[data_row][data_col] = out_data[data_row][data_col]
							+ data[data_row][diff_index + kn]
									* kernel[kn][k_index];
				}
			}
	}
}

//caculate the grad_convolution for diff
//data : k
//top_diff : diffs
//out_data : diff_prevs
extern "C" void CACU_DECONV_DIFF_GPU(float_t **&data, float_t **&top_diff,
		int kernel_size, int kernels_num, int num, int input_dim, int pad,
		int channel, int stride, float_t **&out_data) {

	int input_dim_ = (input_dim + 2 * pad);
	int output_dim = (input_dim_ - kernel_size) / stride + 1;

	int length = input_dim_ * input_dim_ * channel;

	_k_CACU_DECONV_DIFF_GPU<<<BLOCKNUM, THREADNUM, 0>>>(top_diff, data, num,
			channel, kernels_num, input_dim_, output_dim, stride, kernel_size,
			length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DECONV_DIFF_COL_GPU(float_t *data, float_t *kernel,
		int num, int kernels_num, int block_size, int length,
		float_t *out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

//outset is the index in output feature map
//blockset is the index in block
	int outset, blockset;

	int data_row, data_col;

	int data_length = (length / block_size) * kernels_num;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
		out_data[i] = 0.0;
		outset = data_col / block_size;
		blockset = data_col % block_size;

		for (int j = 0; j < kernels_num; j++) {
			out_data[i] += kernel[j * block_size + blockset]
					* data[data_row * data_length + outset * kernels_num + j];
//			if (i == 100)
//				printf("%f,%f,%f\n", kernel[j * block_size + blockset],
//						data[data_row * data_length + outset * kernels_num + j],
//						out_data[i]);
		}

	}
}

//caculate the grad_convolution for diff
//data : k
//top_diff : diffs
//out_data : diff_prevs
extern "C" void CACU_DECONV_DIFF_COL_GPU(float_t *&data, float_t *&top_diff,
		int kernel_size, int kernels_num, int num, int input_dim, int pad,
		int channel, int stride, float_t *&out_data) {

	int input_dim_ = (input_dim + 2 * pad);
	int output_dim = (input_dim_ - kernel_size) / stride + 1;

	int block_size = kernel_size * kernel_size * channel;

	int length = output_dim * output_dim * channel * kernel_size * kernel_size;

	_k_CACU_DECONV_DIFF_COL_GPU<<<BLOCKNUM, THREADNUM, 0>>>(top_diff, data, num,
			kernels_num, block_size, length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_ACTIVATION_RELU_GPU(float_t **data, int num,
		int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		data[data_row][data_col] = max((float_t) 0, data[data_row][data_col]);

	}
}

extern "C" void CACU_ACTIVATION_RELU_GPU(float_t **&data, int num, int length) {

	_k_CACU_ACTIVATION_RELU_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_ACTIVATION_LEAKY_RELU_GPU(float_t **data, int num,
		int length, float_t slope) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		data[data_row][data_col] =
				0 <= data[data_row][data_col] ?
						data[data_row][data_col] :
						data[data_row][data_col] * slope;

	}
}

extern "C" void CACU_ACTIVATION_LEAKY_RELU_GPU(float_t **&data, int num,
		int length, float_t slope) {

	_k_CACU_ACTIVATION_LEAKY_RELU_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num,
			length, slope);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_DE_ACTIVATION_RELU_GPU(float_t **data, int num,
		int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	float_t sign;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		sign = data[data_row][data_col] > 0 ? (float_t) 1 : (float_t) 0;
		out_data[data_row][data_col] = sign * out_data[data_row][data_col];
	}
}

extern "C" void CACU_DE_ACTIVATION_RELU_GPU(float_t **&data, int num,
		int length, float_t **&out_data) {

	_k_CACU_DE_ACTIVATION_RELU_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num,
			length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DE_ACTIVATION_LEAKY_RELU_GPU(float_t **data, int num,
		int length, float_t slope, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	float_t sign;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		sign = data[data_row][data_col] > 0 ? (float_t) 1 : slope;
		out_data[data_row][data_col] = sign * out_data[data_row][data_col];
	}
}

extern "C" void CACU_DE_ACTIVATION_LEAKY_RELU_GPU(float_t **&data, int num,
		int length, float_t slope, float_t **&out_data) {

	_k_CACU_DE_ACTIVATION_LEAKY_RELU_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num,
			length, slope, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_ACTIVATION_SIGMOID_GPU(float_t **data, int num,
		int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		data[data_row][data_col] = float_t(1)
				/ (float_t(1) + exp(-data[data_row][data_col]));
	}
}

extern "C" void CACU_ACTIVATION_SIGMOID_GPU(float_t **&data, int num,
		int length) {

	_k_CACU_ACTIVATION_SIGMOID_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num,
			length);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DE_ACTIVATION_SIGMOID_GPU(float_t **data, int num,
		int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		out_data[data_row][data_col] = data[data_row][data_col]
				* (float_t(1) - data[data_row][data_col]);
	}
}

extern "C" void CACU_DE_ACTIVATION_SIGMOID_GPU(float_t **&data, int num,
		int length, float_t **&out_data) {

	_k_CACU_DE_ACTIVATION_SIGMOID_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num,
			length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SOFTMAX_GPU(float_t **data, int num, int length,
		float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	__shared__ float_t sum, max_data;

	for (int j = bid; j < num; j += BLOCKNUM) {

		if (tid == 0) {
			max_data = data[bid][0];
			for (int i = 1; i < length; i++)
				max_data = max(max_data, data[bid][i]);
		}

		__syncthreads();

		for (int i = tid; i < length; i += THREADNUM) {
			data[bid][i] = exp(data[bid][i] - max_data);
		}

		__syncthreads();

		if (tid == 0) {
			sum = 0;
			for (int i = 0; i < length; i++)
				sum += data[bid][i];
		}

		__syncthreads();

		for (int i = tid; i < length; i += THREADNUM) {
			out_data[bid][i] = data[bid][i] / sum;
		}
	}
}

extern "C" void CACU_SOFTMAX_GPU(float_t **&data, int num, int length,
		float_t **&out_data) {

	_k_CACU_SOFTMAX_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length,
			out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_GEMM_GPU(float_t **data, float_t **kernel,
		float_t **bias, int num, int kernels_num, int length,
		float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * kernels_num; i += BLOCKNUM * THREADNUM) {
		data_row = i / kernels_num;
		data_col = i % kernels_num;

		out_data[data_row][data_col] = 0.0;

		for (int j = 0; j < length; j++) {
			out_data[data_row][data_col] = out_data[data_row][data_col]
					+ data[data_row][j] * kernel[data_col][j];
		}
		out_data[data_row][data_col] = out_data[data_row][data_col]
				+ bias[data_col][0];
	}
}

//caculate the sum(a*x_0i+b)
extern "C" void CACU_GEMM_GPU(float_t **&data, float_t **&bias, int num,
		int kernels_num, int length, float_t **&kernels, float_t **&out_data) {

	_k_CACU_GEMM_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, kernels, bias, num,
			kernels_num, length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DE_GEMM_W_GPU(float_t **data, float_t **scales, int num,
		int kernels_num, int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < length * kernels_num; i +=
			BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;

		out_data[data_row][data_col] = 0.0;

		for (int j = 0; j < num; j++) {
			out_data[data_row][data_col] = out_data[data_row][data_col]
					+ data[j][data_row] * scales[j][data_col];
		}
	}
}

//data : top_diff
//scales : bottoms_data
//out_data : grad for w
extern "C" void CACU_DE_GEMM_W_GPU(float_t **&data, int num, int kernels_num,
		int length, float_t **&scales, float_t **&out_data) {

	_k_CACU_DE_GEMM_W_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, scales, num,
			kernels_num, length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_DE_GEMM_DIFF_GPU(float_t **data, float_t **scales,
		int num, int kernels_num, int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < length * num; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;

		out_data[data_row][data_col] = 0.0;

		for (int j = 0; j < kernels_num; j++) {
			out_data[data_row][data_col] = out_data[data_row][data_col]
					+ data[data_row][j] * scales[j][data_col];
		}
	}
}

//data : top_diff
//scales : w
//out_data : bottoms_diff
extern "C" void CACU_DE_GEMM_DIFF_GPU(float_t **&data, int num, int kernels_num,
		int length, float_t **&scales, float_t **&out_data) {

	_k_CACU_DE_GEMM_DIFF_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, scales, num,
			kernels_num, length, out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_AXBY_GPU(float_t **data, float_t a, float_t **bias,
		float_t b, int num, int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		out_data[data_row][data_col] = data[data_row][data_col] * a
				+ bias[data_row][data_col] * b;
	}
}

//caculate the sum(a*x_0i+by)
extern "C" void CACU_AXBY_GPU(float_t **&data, float_t a, int num, int length,
		float_t **&bias, float_t b, float_t **&out_data) {

	_k_CACU_AXBY_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, a, bias, b, num, length,
			out_data);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_AXBY_CROP_GPU(float_t **data, float_t a, float_t **bias,
		float_t b, int num, int length, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;

		if (abs(data[data_row][data_col] * a + bias[data_row][data_col] * b)
				< 1)
			out_data[data_row][data_col] = data[data_row][data_col] * a
					+ bias[data_row][data_col] * b;
		else
			out_data[data_row][data_col] = data[data_row][data_col];
	}
}

//caculate ||r|| < 1
extern "C" void CACU_AXBY_CROP_GPU(float_t **&data, float_t a, int num,
		int length, float_t **&bias, float_t b, float_t **&out_data) {

	_k_CACU_AXBY_CROP_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, a, bias, b, num,
			length, out_data);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_A_POOLING_GPU(float_t **data, int num, int kernel_size,
		int input_dim, int output_dim, int pad, int out_length, int channel,
		int stride, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int set_i, set_j;

	int start_i, start_j;

	int start_in;

	int c;

	int data_row, data_col;

	float_t sum;
	int count;

	for (int i = threadid; i < num * out_length; i += BLOCKNUM * THREADNUM) {

		data_row = i / out_length;
		data_col = i % out_length;

		sum = 0;
		count = 0;

		set_i = (data_col / channel) / output_dim;
		set_j = (data_col / channel) % output_dim;

		start_i = set_i * stride;
		start_j = set_j * stride;

		c = data_col % channel;

		start_in = (start_i * input_dim + start_j) * channel + c;

		for (int ki = 0; ki < kernel_size && (ki + start_i) < input_dim; ki++) {
			for (int kj = 0; kj < kernel_size && (kj + start_j) < input_dim;
					kj++) {
				sum +=
						data[data_row][start_in
								+ (ki * input_dim + kj) * channel];
				count++;
			}
		}
		out_data[data_row][data_col] = (float_t) (sum / count);
	}
}

//caculate the sum(a*x_0i+b)
extern "C" void CACU_A_POOLING_GPU(float_t **&data, int num, int kernel_size,
		int input_dim, int output_dim, int pad, int out_length, int channel,
		int stride, float_t **&out_data) {

	_k_CACU_A_POOLING_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, kernel_size,
			input_dim, output_dim, pad, out_length, channel, stride, out_data);

	cudaThreadSynchronize();

}

__global__ void _k_CACU_M_POOLING_GPU(float_t **data, int num, int kernel_size,
		int input_dim, int output_dim, int out_length, int channel, int stride,
		float_t **out_data, float_t **index) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int set_i, set_j;

	int start_i, start_j;

	int start_in;

	int c;

	int data_row, data_col;

	float_t sign;

	for (int i = threadid; i < num * out_length; i += BLOCKNUM * THREADNUM) {

		data_row = i / out_length;
		data_col = i % out_length;

		set_i = (data_col / channel) / output_dim;
		set_j = (data_col / channel) % output_dim;

		start_i = set_i * stride;
		start_j = set_j * stride;

		c = data_col % channel;

		start_in = (start_i * input_dim + start_j) * channel + c;

		for (int ki = 0; ki < kernel_size && (ki + set_i * stride) < input_dim;
				ki++)
			for (int kj = 0;
					kj < kernel_size && (kj + set_j * stride) < input_dim;
					kj++) {
				sign =
						data[data_row][start_in
								+ (ki * input_dim + kj) * channel];
				if (out_data[data_row][data_col] < sign
						|| (ki == 0 && kj == 0)) {
					index[data_row][data_col] = ki * kernel_size + kj;
					out_data[data_row][data_col] = sign;
				}
			}
	}
}

//caculate the sum(a*x_0i+b)
extern "C" void CACU_M_POOLING_GPU(float_t **&data, int num, int kernel_size,
		int input_dim, int output_dim, int out_length, int channel, int stride,
		float_t **&out_data, float_t **index) {

	_k_CACU_M_POOLING_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, kernel_size,
			input_dim, output_dim, out_length, channel, stride, out_data,
			index);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_CE_LOSS_GPU(float_t **data, float_t **label, int num,
		float_t *loss) {

	int tid = threadIdx.x;

	loss[0] = 0;

	__shared__ float_t share_data[THREADNUM];

	share_data[tid] = 0;

	for (int i = tid; i < num; i += THREADNUM) {

		int index = int(label[i][0]);

		share_data[tid] -= (log(data[i][index]));
	}

	__syncthreads();

	if (tid == 0) {
		for (int i = 1; i < THREADNUM; i++)
			share_data[0] += share_data[i];

		loss[0] = share_data[0];
	}

}

//caculate the loss
extern "C" void CACU_CE_LOSS_GPU(float_t **&data, float_t **label, int num,
		float_t *&loss) {

	_k_CACU_CE_LOSS_GPU<<<1, THREADNUM, 0>>>(data, label, num, loss);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_SUB_INDEX_GPU(float_t **data, float_t **label, int num,
		float_t value, float_t **out_data) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < num; i += BLOCKNUM * THREADNUM) {

		int index = int(label[i][0]);

		out_data[i][index] -= value;
	}

}

//caculate the loss
extern "C" void CACU_SUB_INDEX_GPU(float_t **&data, float_t ** index,
		float_t value, int num, float_t **&out_data) {

	_k_CACU_SUB_INDEX_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, index, num, value,
			out_data);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_RESET_DATA_GPU(float_t **data_input, int num,
		int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int out_start;

	int data_row;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_row = j / length;
		out_start = j % length;
		data_input[data_row][out_start] = 0;
	}
}

extern "C" void CACU_RESET_DATA_GPU(float_t **&data, int num, int length) {

	_k_CACU_RESET_DATA_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length);

	cudaThreadSynchronize();
}

__global__ void _k_CACU_RESET_BIN_DATA_GPU(unsigned int **data_input, int num,
		int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int out_start;

	int data_row;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_row = j / length;
		out_start = j % length;
		data_input[data_row][out_start] = 0;
	}
}

extern "C" void CACU_RESET_BIN_DATA_GPU(unsigned int **&data, int num,
		int length) {

	_k_CACU_RESET_BIN_DATA_GPU<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length);

	cudaThreadSynchronize();

}
