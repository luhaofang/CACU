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

#define BLOCKNUM 512
#define THREADNUM 512

__global__ void _k_copy_padding_data_blob_gpu(float_t *data_input,
		float_t *data_output, int num, int input_dim, int channel, int pad) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int output_dim = input_dim + pad * 2;

	int output_length = output_dim * output_dim * channel;

	int in_start, row, col;

	int data_row, data_col;
	int indata_length = input_dim * input_dim * channel;
	for (int j = threadid; j < num * output_length; j += BLOCKNUM * THREADNUM) {
		data_row = j / output_length;
		data_col = j % output_length;
		row = data_col / (output_dim * channel);
		//col = (data_col % (output_dim * channel)) / channel;
		col = (data_col / channel) % output_dim;
		if (row >= pad && row < output_dim - pad) {
			if (col >= pad && col < output_dim - pad) {
				in_start = ((row - pad) * input_dim + (col - pad)) * channel
						+ data_col % channel;
				data_output[j] =
						data_input[data_row * indata_length + in_start];
			} else
				data_output[j] = 0.0;
		} else
			data_output[j] = 0.0;
	}
}

extern "C" void copy_padding_data_blob_gpu(float_t *&data, int num,
		int input_dim, int channel, int pad, float_t *&out_data) {

	_k_copy_padding_data_blob_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data,
			num, input_dim, channel, pad);

	cudaThreadSynchronize();
}

__global__ void _k_append_padding_data_blob_gpu(float_t **data_input,
		float_t **data_output, int num, int input_dim, int channel, int pad) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int output_dim = input_dim + pad;

	int output_length = output_dim * output_dim * channel;

	int out_start, in_start, row, col;

	int data_row, data_col;
	for (int j = threadid; j < num * output_length; j += BLOCKNUM * THREADNUM) {

		data_row = j / output_length;
		data_col = j % output_length;
		out_start = data_col;
		row = data_col / (output_dim * channel);
		//col = (data_col % (output_dim * channel)) / channel;
		col = (data_col / channel) % output_dim;
		if (row < output_dim - pad) {
			if (col < output_dim - pad) {
				in_start = ((row) * input_dim + col) * channel
						+ data_col % channel;
				data_output[data_row][out_start] =
						data_input[data_row][in_start];
			} else
				data_output[data_row][out_start] = 0.0;
		} else
			data_output[data_row][out_start] = 0.0;
	}
}

extern "C" void append_padding_data_blob_gpu(float_t **&data, int num,
		int input_dim, int channel, int pad, float_t **&out_data) {

	_k_append_padding_data_blob_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data,
			num, input_dim, channel, pad);

	cudaThreadSynchronize();

}

__global__ void _k_copy_unpadding_data_gpu(float_t *data_input,
		float_t *data_output, int num, int input_dim, int channel, int pad) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int length = input_dim * input_dim * channel;

	int output_dim = input_dim + 2 * pad;

	int indata_length = output_dim*output_dim*channel;

	int in_start, row, col;

	int data_row, data_col;
	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_row = j / length;
		data_col = j % length;
		row = data_col / (input_dim * channel);
		//col = (data_col % (input_dim * channel)) / channel;
		col = (data_col / channel) % input_dim;
		in_start = ((row + pad) * output_dim + (col + pad)) * channel
				+ data_col % channel;
		data_output[j] = data_input[data_row * indata_length + in_start];
	}
}

extern "C" void copy_unpadding_data_gpu(float_t *&data, int num, int input_dim,
		int channel, int pad, float_t *&out_data) {

	_k_copy_unpadding_data_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data, num,
			input_dim, channel, pad);

	cudaThreadSynchronize();
}

__global__ void _k_append_unpadding_data_gpu(float_t **data_input,
		float_t **data_output, int num, int input_dim, int channel, int pad) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int output_length = input_dim * input_dim * channel;

	int output_dim = input_dim + pad;

	int out_start, in_start, row, col;

	int data_row, data_col;

	for (int j = threadid; j < num * output_length; j += BLOCKNUM * THREADNUM) {
		data_row = j / output_length;
		data_col = j % output_length;
		out_start = data_col;
		row = data_col / (input_dim * channel);
		//col =(data_col % (input_dim * channel)) / channel;
		col = (data_col / channel) % input_dim;
		in_start = ((row) * output_dim + (col)) * channel + data_col % channel;
		data_output[data_row][out_start] = data_input[data_row][in_start];
	}
}

extern "C" void append_unpadding_data_gpu(float_t **&data, int num,
		int input_dim, int channel, int pad, float_t **&out_data) {

	_k_append_unpadding_data_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data,
			num, input_dim, channel, pad);

	cudaThreadSynchronize();

}

__global__ void _k_copy_padding_data_sign_gpu(unsigned int *data_input,
		unsigned int *data_output, int num, int input_dim, int channel,
		int pad) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int output_dim = input_dim + pad * 2;

	int output_length = output_dim * output_dim * channel;

	int input_length = input_dim * input_dim * channel;

	int in_start, row, col;

	int data_row, data_col;
	for (int j = threadid; j < num * output_length; j += BLOCKNUM * THREADNUM) {
		data_row = j / output_length;
		data_col = j % output_length;
		row = data_col / (output_dim * channel);
		//col = (data_col % (output_dim * channel)) / channel;
		col = (data_col / channel) % output_dim;
		if (row >= pad && row < output_dim - pad) {
			if (col >= pad && col < output_dim - pad) {
				in_start = ((row - pad) * input_dim + (col - pad)) * channel
						+ data_col % channel;
				data_output[j] = data_input[data_row * input_length + in_start];
			} else
				data_output[j] = 0;
		} else
			data_output[j] = 0;
	}
}

extern "C" void copy_padding_data_sign_gpu(unsigned int *&data, int num,
		int input_dim, int channel, int pad, unsigned int *&out_data) {

	_k_copy_padding_data_sign_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data,
			num, input_dim, channel, pad);

	cudaThreadSynchronize();

}

__global__ void _k_img2col_gpu(float_t *data_input, float_t *data_output,
		int num, int block_size, int output_length, int channel, int input_dim,
		int output_dim, int stride, int kernel_size) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int border = input_dim - output_dim;

	int out_start, in_start, in, out;

	int data_row, data_col;

	int indata_length = input_dim * input_dim * channel;
	int outdata_length = output_length * block_size * channel;

	for (int j = threadid; j < num * output_length; j += BLOCKNUM * THREADNUM) {
		data_row = j / output_length;
		data_col = j % output_length;
		out_start = data_col * (block_size * channel);
		in_start = (data_col + (data_col / output_dim) * border) * channel;
		for (int c = 0; c < channel; c++) {
			for (int ki = 0; ki < kernel_size; ki++) {
				for (int kj = 0; kj < kernel_size; kj++) {
					in = in_start + (ki * input_dim + kj) * channel + c;
					out = out_start + c * block_size + ki * kernel_size + kj;
					data_output[data_row * outdata_length + out] =
							data_input[data_row * indata_length + in];
				}
			}
		}
	}
}

extern "C" void img2col_gpu(float_t *&data, int num, int channel, int input_dim,
		int kernel_size, int stride, int output_dim, float_t *&pad_input) {

	int block_size = kernel_size * kernel_size;
	int output_length = output_dim * output_dim;

	_k_img2col_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, pad_input, num, block_size,
			output_length, channel, input_dim, output_dim, stride, kernel_size);

	cudaThreadSynchronize();
}

__global__ void _k_col2img_gpu(float_t *data, int num, int channel,
		int input_dim, int output_dim, int stride, int kernel_size, int length,
		float_t *out_data) {

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

	int k_index, outset_index;

	int block_size = kernel_size * kernel_size * channel;

	int indata_length = output_dim*output_dim*block_size;

	int c;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

		out_data[i] = 0.0;

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
						+ (startset_j - outset_j * stride))
						+ c * kernel_size * kernel_size;
				outset_index = (outset_i * output_dim + outset_j) * block_size;

				out_data[i] += data[data_row*indata_length + outset_index + k_index];

			}
	}
}

extern "C" void col2img_gpu(float_t *&data, int num, int channel,
		int input_dim, int kernel_size, int stride, int output_dim,
		float_t *&pad_input) {

	int length = input_dim * input_dim * channel;

	_k_col2img_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, num, channel, input_dim,
			output_dim, stride, kernel_size, length, pad_input);

	cudaThreadSynchronize();
}

__global__ void _k_img2bitcol_gpu(unsigned int *data_input,
		unsigned int *data_output, int num, int block_size, int output_length,
		int channel, int input_dim, int output_dim, int stride, int length,
		int kernel_size) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int border = input_dim - output_dim;
	int sp[BIN_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned int R[BIN_SIZE] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
			2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
			1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864,
			134217728, 268435456, 536870912, 1073741824, 2147483648 };
	int end_flag = kernel_size * kernel_size * channel - 1;
	int count = 0, index, out_start, in_start, in;
	int data_row, data_col;
	unsigned int data = 0;
	int outdata_length = output_length * block_size;
	int indata_length = input_dim * input_dim * channel;
	for (int j = threadid; j < num * output_length; j += BLOCKNUM * THREADNUM) {
		data_row = j / output_length;
		data_col = j % output_length;
		out_start = data_col * block_size;
		in_start = (data_col + (data_col / output_dim) * border) * channel;
		count = 0;
		for (int c = 0; c < channel; c++) {
			for (int ki = 0; ki < kernel_size; ki++) {
				for (int kj = 0; kj < kernel_size; kj++) {
					in = in_start + (ki * input_dim + kj) * channel + c;
					index = count % BIN_SIZE;
					sp[index] = data_input[data_row * indata_length + in];
					if (index == BIN_SIZE - 1 || count == end_flag) {
						for (int i = 0; i < BIN_SIZE; i++) {
							data += R[i] * sp[i];
						}
						data_output[data_row * outdata_length + out_start] =
								data;
						data = 0;
						out_start += 1;
						for (int m = 0; m < BIN_SIZE; m++)
							sp[m] = 0;
					}
					count++;
				}
			}
		}
	}
}

extern "C" void img2bitcol_gpu(unsigned int *&bin_data, int num, int channel,
		int input_dim, int kernel_size, int stride, int pad, int output_dim,
		unsigned int *&pad_input) {

	clock_t start = clock();

	int length;

	if (channel * kernel_size * kernel_size % BIN_SIZE == 0)
		length = (channel * kernel_size * kernel_size / BIN_SIZE) * output_dim
				* output_dim;
	else
		length = (channel * kernel_size * kernel_size / BIN_SIZE + 1)
				* output_dim * output_dim;

	int block_size = length / (output_dim * output_dim);
	int output_length = output_dim * output_dim;
	int input_dim_ = input_dim + 2 * pad;

	_k_img2bitcol_gpu<<<BLOCKNUM, THREADNUM, 0>>>(bin_data, pad_input, num,
			block_size, output_length, channel, input_dim_, output_dim, stride,length,
			kernel_size);

	cudaThreadSynchronize();
}

__global__ void _k_copy_data_gpu(float_t **data_input, float_t **data_output,
		int num, int length, int add) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int out_start, in_start;

	int data_row;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_row = j / length;
		out_start = j % length;
		in_start = j % length;
		if (add) {
			data_output[data_row][out_start] += data_input[data_row][in_start];
		} else {
			data_output[data_row][out_start] = data_input[data_row][in_start];
		}
	}
}

extern "C" void copy_data_gpu(float_t **&data, float_t **&out_data, int num,
		int length, int add) {

	_k_copy_data_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data, num, length,
			add);

	cudaThreadSynchronize();
}

__global__ void _k_copy_data_bin_gpu(unsigned int **data_input,
		unsigned int **data_output, int num, int length, int add) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int out_start, in_start;

	int data_row;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_row = j / length;
		out_start = j % length;
		in_start = j % length;
		if (add) {
			data_output[data_row][out_start] += data_input[data_row][in_start];
		} else {
			data_output[data_row][out_start] = data_input[data_row][in_start];
		}
	}
}

extern "C" void copy_data_bin_gpu(unsigned int **&data,
		unsigned int **&out_data, int num, int length, int add) {

	_k_copy_data_bin_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data, num,
			length, add);

	cudaThreadSynchronize();
}

__global__ void _k_copy2dest_gpu(float_t **data_input, float_t **index_data,
		float_t **data_output, int num, int input_dim, int output_dim,
		int channel, int kernel_size, int stride, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;
	//the index for the data in kernel
	int offset_i, offset_j;

	int c;
	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {
		data_row = i / length;
		data_col = i % length;
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

		for (int mi = 0; mi <= count_i; mi++)
			for (int mj = 0; mj <= count_j; mj++) {
				outset_i = outset_si - mi;
				outset_j = outset_sj - mj;

				offset_i = startset_i - outset_i * stride;
				offset_j = startset_j - outset_j * stride;
				if (index_data[data_row][(outset_i * output_dim + outset_j)
						* channel + c]
						== (float_t) (offset_i * kernel_size + offset_j)) {
					data_output[data_row][data_col] +=
							data_input[data_row][(outset_i * output_dim
									+ outset_j) * channel + c];
				}
			}

	}
}

extern "C" void copy2dest_gpu(float_t **&data, float_t **&index_data, int num,
		int output_dim, int input_dim, int channel, int kernel_size, int stride,
		float_t **&out_data) {

	int length = input_dim * input_dim * channel;

	_k_copy2dest_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, index_data, out_data,
			num, input_dim, output_dim, channel, kernel_size, stride, length);

	cudaThreadSynchronize();
}

__global__ void _k_copy2mean_gpu(float_t **data_input, float_t **data_output,
		int num, int channel, int input_dim, int output_dim, int stride,
		int kernel_size, int pad, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;

	int pw, ph;

	int c;

	int data_row, data_col;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		data_row = i / length;
		data_col = i % length;

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

				pw = kernel_size;
				ph = kernel_size;

				if (outset_i == output_dim - 1)
					ph = kernel_size - pad;

				if (outset_j == output_dim - 1)
					pw = kernel_size - pad;

				data_output[data_row][data_col] +=
						(data_input[data_row][(outset_i * output_dim + outset_j)
								* channel + c] / (float_t) (ph * pw));
			}
	}
}

extern "C" void copy2mean_gpu(float_t **&data, int num, int output_dim,
		int input_dim, int channel, int kernel_size, int stride, int pad,
		float_t **&out_data) {

	int length = input_dim * input_dim * channel;

	_k_copy2mean_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, out_data, num, channel,
			input_dim, output_dim, stride, kernel_size, pad, length);

	cudaThreadSynchronize();

}

__global__ void _k_reset_data_gpu(float_t *data_input, int num, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_input[j] = 0;
	}
}

extern "C" void reset_data_gpu(float_t *&data, int num, int length) {

	_k_reset_data_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length);

	cudaThreadSynchronize();
}

__global__ void _k_reset_bin_data_gpu(unsigned int *data_input, int num,
		int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_input[j] = 0;
	}
}

extern "C" void reset_bin_data_gpu(unsigned int *&data, int num, int length) {

	_k_reset_bin_data_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length);

	cudaThreadSynchronize();
}

__global__ void _k_set_data_gpu(float_t **data_input, int num, int length,
		float_t value) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int out_start;

	int data_row;

	for (int j = threadid; j < num * length; j += BLOCKNUM * THREADNUM) {
		data_row = j / length;
		out_start = j % length;

		data_input[data_row][out_start] = value;

	}
}

extern "C" void set_data_gpu(float_t **&data, int num, int length,
		float_t value) {

	_k_set_data_gpu<<<BLOCKNUM, THREADNUM, 0>>>(data, num, length, value);

	cudaThreadSynchronize();
}

