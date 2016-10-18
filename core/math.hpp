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

#include "../activation_fuction.h"

namespace mycnn {

#if GPU_MODE

//vec_t(size) -> vec_t(size/sum_size)
extern "C" void CACU_SUM_SIZE_GPU(float_t **&data, int num, int sum_size,
		int length, int out_length, float_t **&out_data);

extern "C" void CACU_MEAN_GPU(float_t **&data, int num, int length,
		float_t **&out_data);

//vec_t(size) -> vec_t(size/sum_size)
extern "C" void CACU_SUM_SIZE_ABS_GPU(float_t **&data, int num, int sum_size,
		int length, int out_length, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the means for batch_size
extern "C" void CACU_MEAN_CHANNEL_GPU(float_t **&data, int num, int length,
		int channel, float_t *&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the variance for batch_size
extern "C" void CACU_VARIANCE_CHANNEL_GPU(float_t **&data, float_t *&mean,
		int num, int length, int channel, float_t *&out_data);

extern "C" void CACU_DOT_GPU(float_t **&data, float_t **&scale, int num,
		int length, float_t **&out_data);

extern "C" void CACU_SQRT_GPU(float_t **&data, int num, int length,
		float_t **&out_data);

extern "C" void CACU_SCALE_GPU(float_t **&data, float_t *&scale, int num,
		int length, int channel, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix A*B
extern "C" void CACU_SCALE_GPU_D(float_t **&data, float_t **&scale, int num,
		int length, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix scale*B
extern "C" void CACU_SCALE_GPU_A(float_t **&data, float_t scale, int num,
		int length, float_t **&out_data, int add);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel' scale_sum  for batch_size
extern "C" void CACU_SCALE_GPU_B(float_t **&data, float_t **&scale, int num,
		int length, int channel, float_t *&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum bias for batch_size
extern "C" void CACU_SUM_GPU(float_t **&data, float_t *&bias, int num,
		int length,int channel, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum for batch_size
extern "C" void CACU_SUM_GPU_B(float_t **&data, int num, int length,
		int channel, float_t *&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum for every sample
extern "C" void CACU_SUM_GPU_C(float_t **&data, int num, int length,
		int out_length, int channel, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum(abs(x)) for every sample
extern "C" void CACU_SUM_ABS_GPU(float_t **&data, int num, int length,
		int out_length, int channel, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the dim's sum for every batch_size
extern "C" void CACU_SUM_GPU_D(float_t **&data, float_t **&bias, int num,
		int length, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum bias for batch_size
extern "C" void CACU_SUM_GPU_R(float_t **&data, float_t **&bias, int num,
		int output_channel, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the subtraction for batch_size
extern "C" void CACU_SUB_GPU(float_t **&data, float_t *&bias, int num,
		int length, int channel, float_t **&out_data);

extern "C" void CACU_SUB_GPU_D(float_t **&data, float_t **&bias, int num,
		int length, float_t **&out_data);

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the division for batch_size
extern "C" void CACU_DIVISION_GPU(float_t **&data, float_t *&scale, int num,
		int length, int channel, float_t **&out_data);

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
extern "C" void CACU_ROU_GPU(float_t **&data, float_t **&dx_ba, float_t *&mean,
		float_t *&variance, int num, int length, int channel,
		float_t *&out_data);

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
extern "C" void CACU_MU_GPU(float_t **&data, float_t **&dx_ba, float_t *&mean,
		float_t *&variance, float_t *&rou, int num, int length, int channel,
		float_t *&out_data);

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
extern "C" void CACU_DX_GPU(float_t **&data, float_t **&dx_ba, float_t *&mean,
		float_t *&variance, float_t *&rou, float_t *&mu, int num, int length,
		int channel, float_t **&out_data);

//caculate the sum(a*x_0i)
extern "C" void CACU_SCALE_SUM_ROW_GPU(float_t **&data, int num, int sum_size,
		int kernels_num, int out_length, float_t **&kernels, float_t **&bias,
		float_t **&out_data);

//caculate the grad_convolution for W
//data : bottom
//top_diff : diffs
//out_data : diff_ws
extern "C" void CACU_DECONV_W_BIN_GPU(float_t **&data, float_t **&top_diff, float_t **a,int num,
		int kernel_size, int kernels_num, int output_dim, int channel,
		int stride, float_t **&out_data);

//caculate the grad_convolution for W
//data : bottom
//top_diff : diffs
//out_data : diff_ws
extern "C" void CACU_DECONV_W_B_GPU(float_t **&data, float_t **&top_diff,
		int num, int kernel_size, int kernels_num, int output_dim,
		int channel, int stride, float_t **&out_data, float_t **&bias);

//caculate the grad_convolution for diff
//data : k
//top_diff : diffs
//out_data : diff_prevs
extern "C" void CACU_DECONV_DIFF_GPU(float_t **&data, float_t **&top_diff,
		int kernel_size, int kernels_num, int num, int input_dim, int pad,
		int channel, int stride, float_t **&out_data);

extern "C" void CACU_DECONV_DIFF_COL_GPU(float_t **&data, float_t **&top_diff,
		int kernel_size, int kernels_num, int num, int input_dim, int pad,
		int channel, int stride, float_t **&out_data);

//data : top_diff
//scales : bottoms_data
//out_data : grad for w
extern "C" void CACU_DE_GEMM_W_GPU(float_t **&data, int num, int kernels_num,
		int length, float_t **&scales, float_t **&out_data);

//data : top_diff
//scales : w
//out_data : bottoms_diff
extern "C" void CACU_DE_GEMM_DIFF_GPU(float_t **&data, int num, int kernels_num,
		int length, float_t **&scales, float_t **&out_data);

extern "C" void CACU_ACTIVATION_RELU_GPU(float_t **&data, int num, int length);

extern "C" void CACU_DE_ACTIVATION_RELU_GPU(float_t **&data, int num,
		int length, float_t **&out_data);

extern "C" void CACU_ACTIVATION_SIGMOID_GPU(float_t **&data, int num,
		int length);

extern "C" void CACU_DE_ACTIVATION_SIGMOID_GPU(float_t **&data, int num,
		int length, float_t **&out_data);

extern "C" void CACU_SOFTMAX_GPU(float_t **&data, int num, int length,
		float_t **&out_data);

extern "C" void CACU_GEMM_GPU(float_t **&data, float_t **&bias, int num,
		int kernels_num, int length, float_t **&kernels,
		float_t **&out_data);

extern "C" void CACU_AXBY_GPU(float_t **&data, float_t a, int num, int length,
		float_t **&bias, float_t b, float_t **&out_data);

//caculate ||r|| < 1
extern "C" void CACU_AXBY_CROP_GPU(float_t **&data, float_t a, int num,
		int length, float_t **&bias, float_t b, float_t **&out_data);

extern "C" void CACU_A_POOLING_GPU(float_t **&data, int num,
		int kernel_size,int input_dim, int output_dim, int pad, int out_length, int channel,int stride,
		float_t **&out_data);

extern "C" void CACU_M_POOLING_GPU(float_t **&data, int num, int kernel_size,
		int input_dim, int output_dim, int out_length, int channel, int stride,
		float_t **&out_data, float_t **index);

extern "C" void CACU_CE_LOSS_GPU(float_t **&data, float_t **label, int num,
		float_t *&loss);

extern "C" void CACU_SUB_INDEX_GPU(float_t **&data, float_t ** index,
		float_t value, int num, float_t **&out_data);

extern "C" void CACU_RESET_DATA_GPU(float_t **&data, int num, int length);

extern "C" void CACU_RESET_BIN_DATA_GPU(unsigned int **&data, int num, int length);

#else

//vec_t(size) -> vec_t(size/sum_size)
void CACU_SUM_SIZE_CPU(vec_t &data, int sum_size, vec_t &out_data) {
	assert(data.size() > 0);
	assert(sum_size <= data.size());
	assert(data.size() % sum_size == 0);

	float_t *sp, *snp;
	sp = &data[0];
	snp = &out_data[0];
	int d_size = data.size();
	int index;
	for (int i = 0; i < d_size; i++) {
		index = i / sum_size;
		*(snp + index) += *(sp + i);
	}
}

//vec_t(size) -> vec_t(size/sum_size)
void CACU_SUM_SIZE_ABS_CPU(vec_t &data, int sum_size, vec_t &out_data) {
	assert(data.size() > 0);
	assert(sum_size <= data.size());
	assert(data.size() % sum_size == 0);

	float_t *sp, *snp;
	sp = &data[0];
	snp = &out_data[0];
	int index;
	int d_size = data.size();
	for (int i = 0; i < d_size; i++) {
		index = i / sum_size;
		*(snp + index) += abs(*(sp + i));
	}

}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the means for batch_size
void CACU_MEAN_CHANNEL_CPU(vector<vec_t> &data, int channel, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data[0].size() % channel == 0);
	assert(out_data.size() == channel);

	int dim = data[0].size() / channel;

	float_t denominator = (float_t) dim * data.size();

	float_t *sp, *snp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				sp = &data[num][0] + f * channel + c;
				*snp = *snp + *sp; // / denominator);
			}
		}
	}
	snp = &out_data[0];
	for (int i = 0; i < channel; i++) {
		*(snp + i) /= denominator;
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the variance for batch_size
void CACU_VARIANCE_CHANNEL_CPU(vector<vec_t> &data, vec_t &mean, int channel,
		vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data[0].size() % channel == 0);
	assert(out_data.size() == channel);

	int dim = data[0].size() / channel;

	out_data.resize(channel, 0.0);

	float_t denominator = (float_t) dim * data.size();

	float_t *sp, *snp, *smp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[0] + c;
			smp = &mean[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				sp = &data[num][0] + f * channel + c;
				*snp = *snp + pow((*sp - *smp), 2);	// / denominator;
			}
		}
	}

	snp = &out_data[0];
	for (int c = 0; c < channel; c++) {
		*(snp + c) = (*(snp + c) / denominator);
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix scale*B
void CACU_DOT(vec_t &data, vec_t &scale, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data.size() == scale.size());

	int dim = data.size();

	float_t *sdp, *snp, *ssp;
	snp = &out_data[0];
	sdp = &data[0];
	ssp = &scale[0];
	//iteration for feature map

	for (int f = 0; f < dim; f++) {
		*(snp + f) = (*(sdp + f) * (*(ssp + f)));
	}

}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix scale*B
void CACU_SQRT(vec_t &data, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());

	int dim = data.size();

	float_t *sdp, *snp;
	snp = &out_data[0];
	sdp = &data[0];
	//iteration for feature map

	for (int f = 0; f < dim; f++) {
		*(snp + f) = sqrt(*(sdp + f));
	}

}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's scale for batch_size
void CACU_SCALE_CPU(vector<vec_t> &data, vec_t &scale, int channel,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	float_t *sp, *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			ssp = &scale[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				sp = &data[num][0] + f * channel + c;
				snp = &out_data[num][0] + f * channel + c;
				*snp = (*sp * (*ssp));
			}
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix A*B
void CACU_SCALE_CPU(vector<vec_t> &data, vector<vec_t> &scale,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() == scale[0].size());

	int dim = data[0].size();

	float_t *sdp, *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		ssp = &scale[num][0];
		snp = &out_data[num][0];
		sdp = &data[num][0];
		//iteration for feature map
		for (int f = 0; f < dim; f++) {
			*(snp + f) = (*(sdp + f) * (*(ssp + f)));
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the matrix scale*B
void CACU_SCALE_CPU(vec_t &data, float_t scale, vec_t &out_data, int add) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());

	int dim = data.size();

	float_t *sdp, *snp;
	snp = &out_data[0];
	sdp = &data[0];
	//iteration for feature map
	if (!add)
		for (int f = 0; f < dim; f++) {
			*(snp + f) = (*(sdp + f) * scale);
		}
	else
		for (int f = 0; f < dim; f++) {
			*(snp + f) += (*(sdp + f) * scale);
		}

}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel' scale_sum  for batch_size
void CACU_SCALE_CPU(vector<vec_t> &data, vector<vec_t> &scale, int channel,
		vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	//assert(data.size() == out_data.size());
	assert(data[0].size() == scale[0].size());

	int dim = data[0].size() / channel;

	int p;

	float_t *sdp, *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		ssp = &scale[num][0];
		sdp = &data[num][0];
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				p = f * channel + c;
				*snp += (*(ssp + p)) * (*(sdp + p));
			}
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum bias for batch_size
void CACU_SUM_CPU(vector<vec_t> &data, vec_t &bias, int channel,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	int p;

	float_t *sdp, *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			ssp = &bias[0] + c;
			sdp = &data[num][0];
			snp = &out_data[num][0];
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				p = f * channel + c;
				*(snp + p) = (*(sdp + p) + (*ssp));
			}
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum for batch_size
void CACU_SUM_CPU(vector<vec_t> &data, int channel, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(channel == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	float_t *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[0] + c;
			ssp = &data[num][0];
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				*snp += *(ssp + f * channel + c);
			}
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum for every sample
void CACU_SUM_CPU(vector<vec_t> &data, int channel, vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() / channel == out_data[0].size());
	assert(data[0].size() % channel == 0);

	int dim = out_data[0].size();

	float_t *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[num][0];
			ssp = &data[num][0];
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				*(snp + f) += *(ssp + f * channel + c);
			}
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the channel's sum(abs(x)) for every sample
void CACU_SUM_ABS_CPU(vector<vec_t> &data, int channel,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() / channel == out_data[0].size());
	assert(data[0].size() % channel == 0);

	int dim = out_data[0].size();

	float_t *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[num][0];
			ssp = &data[num][0];
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				*(snp + f) += abs(*(ssp + f * channel + c));
			}
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the dim's sum for every batch_size
void CACU_SUM_CPU(vector<vec_t> &data, vector<vec_t> &bias,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data[0].size() == out_data[0].size());
	assert(data[0].size() == bias[0].size());
	assert(data[0].size() == out_data[0].size());

	int dim = data[0].size();

	float_t *sdp, *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		snp = &out_data[num][0];
		ssp = &bias[num][0];
		sdp = &data[num][0];
		//iteration for channel
		for (int c = 0; c < dim; c++) {
			*(snp + c) = (*(sdp + c) + *(ssp + c));
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the subtraction for batch_size
void CACU_SUB_CPU(vector<vec_t> &data, vec_t &bias, int channel,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	float_t *sp, *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			ssp = &bias[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				sp = &data[num][0] + f * channel + c;
				snp = &out_data[num][0] + f * channel + c;
				*snp = (*sp - (*ssp));
			}
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the subtraction for batch_size
void CACU_SUB_CPU(vector<vec_t> &data, vector<vec_t> &bias,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());

	float_t *sp, *snp, *ssp;

	int dim = data[0].size();

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		ssp = &bias[num][0];
		//iteration for feature map
		for (int f = 0; f < dim; f++) {
			sp = &data[num][0] + f;
			snp = &out_data[num][0] + f;
			*snp = (*sp - (*ssp));
		}
	}
}

//nums of vec_t(size) -> vec_t(size/sum_size)
//caculate the division for batch_size
void CACU_DIVISION_CPU(vector<vec_t> &data, vec_t &scale, int channel,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	float_t *sp, *snp, *ssp;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			ssp = &scale[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				sp = &data[num][0] + f * channel + c;
				snp = &out_data[num][0] + f * channel + c;
				*snp = (*sp / (*ssp));
			}
		}
	}
}

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
void CACU_ROU_CPU(vector<vec_t> &data, vector<vec_t> &dx_ba, vec_t &mean,
		vec_t &variance, int channel, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	//assert(data.size() == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	float_t denominator = (float_t) dim * data.size();

	float_t *snp, *sxp, *smp, *svp, *sdxp;

	int p;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[0] + c;
			smp = &mean[0] + c;
			svp = &variance[0] + c;
			sdxp = &dx_ba[num][0];
			sxp = &data[num][0];
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				p = f * channel + c;

				*snp += *(sdxp + p) * (*(sxp + p) - *smp)
						* ((float_t) -0.5 / pow(*svp, 3));
			}
		}
	}
}

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
void CACU_MU_CPU(vector<vec_t> &data, vector<vec_t> &dx_ba, vec_t &mean,
		vec_t &variance, vec_t &rou, int channel, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	//assert(data.size() == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	float_t denominator = (float_t) dim * data.size();

	float_t *srp, *snp, *sxp, *smp, *svp, *sdxp;

	int m = dim * data.size();

	int p;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[0] + c;
			smp = &mean[0] + c;
			svp = &variance[0] + c;
			sdxp = &dx_ba[num][0];
			sxp = &data[num][0];
			srp = &rou[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				p = f * channel + c;

				*snp += (*(sdxp + p) / (-*svp))
						+ (*srp) * ((float_t) -2.0 * (*(sxp + p) - *smp) / m);
			}
		}
	}
}

//FOR BATCH_NORMALIZATION not common utilities
//caculate the division for batch_size
void CACU_DX_CPU(vector<vec_t> &data, vector<vec_t> &dx_ba, vec_t &mean,
		vec_t &variance, vec_t &rou, vec_t &mu, int channel,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() % channel == 0);

	int dim = data[0].size() / channel;

	float_t denominator = (float_t) dim * data.size();

	float_t *srp, *snp, *sxp, *smp, *svp, *sdxp, *smup;

	int m = dim * data.size();

	int p;

	for (int num = 0; num < data.size(); num++) {
		//iteration for channel
		for (int c = 0; c < channel; c++) {
			snp = &out_data[num][0];
			smp = &mean[0] + c;
			svp = &variance[0] + c;
			sdxp = &dx_ba[num][0];
			sxp = &data[num][0];
			srp = &rou[0] + c;
			smup = &mu[0] + c;
			//iteration for feature map
			for (int f = 0; f < dim; f++) {
				p = f * channel + c;
				// sum added to bottom diffs
				*(snp + p) += (*(sdxp + p) / (*svp))
						+ (*srp) * ((float_t) 2.0 * (*(sxp + p) - *smp) / m)
						+ (*smup / m);
			}
		}
	}
}

//caculate the sum(x_0i)
void CACU_SUM_ROW_CPU(vector<vec_t> &data, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());

	float_t *snp, *sdp;

	snp = &out_data[0];

	for (int dim = 0; dim < data.size(); dim++) {

		sdp = &data[dim][0];

		for (int d = 0; d < data[0].size(); d++)

			*(snp + dim) += *(sdp + d);
	}
}

//caculate the sum(a*x_0i)
void CACU_SCALE_SUM_ROW_CPU(vec_t &data, int sum_size, vector<vec_t> &kernels,
		vector<vec_t> &bias, vec_t &out_data) {

	assert(data.size() > 0 && out_data.size() > 0 && kernels.size() > 0);
	assert(data.size() / sum_size * kernels.size() == out_data.size());
	assert(kernels[0].size() == sum_size);

	float_t *sp, *snp, *skp;
	sp = &data[0];
	snp = &out_data[0];
	int output_channel = kernels.size();
	int start_index, out_index;
	int d_size = data.size() / sum_size;
	for (int i = 0; i < d_size; i++) {
		start_index = i * sum_size;
		out_index = i * output_channel;
		for (int c = 0; c < output_channel; c++) {
			skp = &kernels[c][0];
			for (int j = 0; j < sum_size; j++) {
				*(snp + out_index + c) += *(sp + start_index + j)
						* (*(skp + j));
			}
			*(snp + out_index + c) += bias[c][0];
		}
	}
}

//caculate the de_convolution for W
//data : bottom
//top_diff : diffs
//out_data : diff_ws
//******************maybe need fixed*******************
void CACU_BIN_DECONV_W_CPU(vec_t &data, vec_t &top_diff, vector<vec_t> real_w,
		vec_t &ks, vector<vec_t> &a, int kernel_size, int input_dim, int pad,
		int channel, int stride, vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0 && top_diff.size() > 0);
	int input_dim_ = (input_dim + 2 * pad);
	int output_dim = (input_dim_ - kernel_size) / stride + 1;
	int output_channel = out_data.size();
	assert(output_dim * output_dim * output_channel == top_diff.size());
	assert(kernel_size * kernel_size * channel == out_data[0].size());
	assert(data.size() == input_dim_ * input_dim_ * channel);

	float_t *sdp, *snp, *sfp, *skp, *srp;
	int sd_out, sf_out;
	float_t *sd_out_cp, *sf_out_cp, *sk_cp, sa;

	activation::sign h;

	sdp = &data[0];
	skp = &ks[0];
	sfp = &top_diff[0];

	//for output_dim's iteration
	for (int i = 0; i < output_dim; i++)
		for (int j = 0; j < output_dim; j++) {
			sd_out = (i * input_dim_ + j) * stride * channel;
			sf_out = (i * output_dim + j) * output_channel;
			sk_cp = skp + (i * output_dim + j);
			//for output_channel's iteration
			for (int co = 0; co < output_channel; co++) {
				sf_out_cp = sfp + sf_out + co;
				snp = &out_data[co][0];
				srp = &real_w[co][0];
				sa = a[co][0];
				//for channel's iteration
				for (int ci = 0; ci < channel; ci++) {
					sd_out_cp = sdp + sd_out + ci;
					//for kernel_size 's iteration
					for (int ki = 0; ki < kernel_size; ki++)
						for (int kj = 0; kj < kernel_size; kj++) {
							//*(snp + (ki * kernel_size + kj) * channel + ci) += h.df(*(srp + (ki * kernel_size + kj) * channel + ci)) * (*(sd_out_cp	+ (ki * input_dim + kj)	* channel) * (*sf_out_cp) * sa * (*sk_cp));
							*(snp + (ki * kernel_size + kj) * channel + ci) +=
									(*(sd_out_cp
											+ (ki * input_dim_ + kj) * channel)
											* (*sf_out_cp));
						}
				}
			}
		}
}

//caculate the de_convolution for diff
//data : real_w
//top_diff : diffs
//out_data : diff_prevs
//******************maybe need fixed*******************
void CACU_BIN_DECONV_DIFF_CPU(vector<vec_t> &data, vec_t &top_diff, vec_t &ks,
		vector<vec_t> &a, int kernel_size, int input_dim, int pad, int channel,
		int stride, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0 && top_diff.size() > 0);
	int input_dim_ = (input_dim + 2 * pad);
	int output_dim = (input_dim_ - kernel_size) / stride + 1;
	int output_channel = data.size();
	assert(output_dim * output_dim * output_channel == top_diff.size());
	assert(kernel_size * kernel_size * channel == data[0].size());

	float_t *sdp, *snp, *sfp, *skp;
	int sf_out, sn_out;
	float_t *sd_out_cp, *sf_out_cp, *sn_out_cp, *sk_cp, sa;

	skp = &ks[0];
	snp = &out_data[0];
	sfp = &top_diff[0];
	//for output_dim's iteration
	for (int i = 0; i < output_dim; i++)
		for (int j = 0; j < output_dim; j++) {
			sn_out = (i * input_dim_ + j) * stride * channel;
			sf_out = (i * output_dim + j) * output_channel;
			sk_cp = skp + (i * output_dim + j);
			//for output_channel's iteration
			for (int co = 0; co < output_channel; co++) {
				sdp = &data[co][0];
				sf_out_cp = sfp + sf_out + co;
				sa = a[co][0];
				//for channel's iteration
				for (int ci = 0; ci < channel; ci++) {
					sd_out_cp = sdp + ci;
					sn_out_cp = snp + sn_out + ci;
					//for kernel_size 's iteration
					for (int ki = 0; ki < kernel_size; ki++)
						for (int kj = 0; kj < kernel_size; kj++) {
							//*(sn_out_cp + (ki * input_dim + kj) * channel) += (*(sd_out_cp+ (ki * kernel_size + kj) * channel)* (*sf_out_cp) * sa * (*sk_cp));
							*(sn_out_cp + (ki * input_dim_ + kj) * channel) +=
									(*(sd_out_cp
											+ (ki * kernel_size + kj) * channel)
											* (*sf_out_cp));
						}
				}
			}
		}
}

//caculate the de_convolution for W
//data : bottom
//top_diff : diffs
//out_data : diff_ws
void CACU_DECONV_W_CPU(vec_t &data, vec_t &top_diff, int kernel_size,
		int input_dim, int pad, int channel, int stride,
		vector<vec_t> &out_data, vector<vec_t> &bias) {
	assert(data.size() > 0 && out_data.size() > 0 && top_diff.size() > 0);
	int input_dim_ = (input_dim + 2 * pad);
	int output_dim = (input_dim_ - kernel_size) / stride + 1;
	int output_channel = out_data.size();
	assert(output_dim * output_dim * output_channel == top_diff.size());
	assert(kernel_size * kernel_size * channel == out_data[0].size());
	assert(
			data.size()
					== output_dim * output_dim * channel * kernel_size
							* kernel_size);

	float_t *sdp, *snp, *sfp;
	int sd_out, sf_out;
	int kernel_length = out_data[0].size();
	float_t *sf_out_cp;

	sdp = &data[0];
	sfp = &top_diff[0];

	//for output_dim's iteration
	for (int i = 0; i < output_dim; i++) {
		for (int j = 0; j < output_dim; j++) {
			sd_out = (i * output_dim + j) * kernel_length;
			sf_out = (i * output_dim + j) * output_channel;
			//for output_channel's iteration
			for (int c = 0; c < output_channel; c++) {
				sf_out_cp = sfp + sf_out + c;
				snp = &out_data[c][0];
				//for channel's iteration
				for (int index = 0; index < kernel_length; index++) {
					*(snp + index) += *(sdp + sd_out + index) * (*sf_out_cp);
				}
				bias[c][0] += *sf_out_cp;
			}
		}
	}
}

//caculate the de_convolution for W
//data : bottom
//top_diff : diffs
//out_data : diff_ws
void CACU_DECONV_W_BIN_CPU(vec_t &data, vec_t &top_diff, int kernel_size,
		int input_dim, int pad, int channel, int stride,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0 && top_diff.size() > 0);
	int input_dim_ = (input_dim + 2 * pad);
	int output_dim = (input_dim_ - kernel_size) / stride + 1;
	int output_channel = out_data.size();
	assert(output_dim * output_dim * output_channel == top_diff.size());
	assert(kernel_size * kernel_size * channel == out_data[0].size());
	assert(
			data.size()
					== output_dim * output_dim * channel * kernel_size
							* kernel_size);

	float_t *sdp, *snp, *sfp;
	int sd_out, sf_out;
	int kernel_length = out_data[0].size();
	float_t *sf_out_cp;

	sdp = &data[0];
	sfp = &top_diff[0];

	float_t crop = 1.0;

	//for output_dim's iteration
	for (int i = 0; i < output_dim; i++) {
		for (int j = 0; j < output_dim; j++) {
			sd_out = (i * output_dim + j) * kernel_size * kernel_size * channel;
			sf_out = (i * output_dim + j) * output_channel;
			//for output_channel's iteration
			for (int co = 0; co < output_channel; co++) {
				sf_out_cp = sfp + sf_out + co;
				snp = &out_data[co][0];
				//for channel's iteration
				for (int index = 0; index < kernel_length; index++) {
					*(snp + index) += *(sdp + sd_out + index) * (*sf_out_cp);
				}
			}
		}
	}
}

void CACU_FIX_GRADIENT_CPU(vector<vec_t> &data, vector<vec_t> &a) {
	assert(data.size() > 0 && a.size() > 0);
	assert(data.size() == a.size());

	float_t *sdp, *sap;

	int kernel_length = data[0].size();

	float_t crop = 1.0;

	for (int c = 0; c < data.size(); c++) {

		sap = &a[c][0];
		sdp = &data[c][0];

		for (int index = 0; index < kernel_length; index++) {
			if (abs(*(sdp + index)) > 1)
				crop = 0.0;
			*(sdp + index) *= ((float_t) (1.0 / kernel_length) + crop * (*sap))
					* (1.0 - (float_t) (1.0 / kernel_length)) * kernel_length;
		}
	}
}

//caculate the de_convolution for diff
//data : real_w
//top_diff : diffs
//out_data : diff_prevs
//******************maybe need fixed*******************
void CACU_DECONV_DIFF_CPU(vector<vec_t> &data, vec_t &top_diff, int kernel_size,
		int channel, int output_dim, vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0 && top_diff.size() > 0);
	int output_channel = data.size();
	assert(
			output_dim * output_dim * channel * kernel_size * kernel_size
					== out_data.size());
	assert(kernel_size * kernel_size * channel == data[0].size());

	float_t *sdp, *snp, *sfp;
	int sf_out, sn_out;
	float_t *sf_out_cp;

	int length = kernel_size * kernel_size * channel;

	snp = &out_data[0];
	sfp = &top_diff[0];
	//for output_dim's iteration
	for (int i = 0; i < output_dim; i++)
		for (int j = 0; j < output_dim; j++) {
			sn_out = (i * output_dim + j) * length;
			sf_out = (i * output_dim + j) * output_channel;
			//for output_channel's iteration
			for (int co = 0; co < output_channel; co++) {
				sdp = &data[co][0];
				sf_out_cp = sfp + sf_out + co;
				//for channel's iteration
				for (int index = 0; index < length; index++) {
					*(snp + sn_out + index) += (*sf_out_cp) * (*(sdp + index));
				}
			}
		}
}

//activation caculation
void CACU_ACTIVATION_RELU_CPU(vector<vec_t> &data) {
	float_t *sp;
	activation::relu relu_;
	int index = 0;
	int length = data[0].size();

	for (int num = 0; num < data.size(); num++) {
		sp = &data[num][0];
		for (int i = 0; i < length; i++) {
			*(sp + i) = relu_.f(*(sp + i));
		}
	}
}

//activation gradient caculation
void CACU_DE_ACTIVATION_RELU_CPU(vector<vec_t> &data, vector<vec_t> &out_data) {
	float_t *sp, *snp;
	activation::relu relu_;
	int index = 0;
	int length = data[0].size();

	for (int num = 0; num < data.size(); num++) {
		sp = &data[num][0];
		snp = &out_data[num][0];
		for (int i = 0; i < length; i++) {

			*(snp + i) = relu_.df(*(sp + i)) * (*(snp + i));
		}
	}
}

//activation caculation
void CACU_ACTIVATION_SIGMOID_CPU(vector<vec_t> &data) {
	float_t *sp;
	activation::sigmoid sigmoid;
	int index = 0;
	int length = data[0].size();

	for (int num = 0; num < data.size(); num++) {
		sp = &data[num][0];
		for (int i = 0; i < length; i++) {
			*(sp + i) = sigmoid.f(*(sp + i));
		}
	}
}

//activation gradient caculation
void CACU_DE_ACTIVATION_SIGMOID_CPU(vector<vec_t> &data,
		vector<vec_t> &out_data) {
	float_t *sp, *snp;
	activation::sigmoid sigmoid;
	int index = 0;
	int length = data[0].size();

	for (int num = 0; num < data.size(); num++) {
		sp = &data[num][0];
		snp = &out_data[num][0];
		for (int i = 0; i < length; i++) {
			//printf("%f * %f =", sigmoid.df(*(sp + i)), *(snp + i));
			*(snp + i) = sigmoid.df(*(sp + i)) * (*(snp + i));
			//printf("%f\n",  *(snp + i));
		}
	}
}

//caculate the sum(a*x_0i+b)
void CACU_GEMM_CPU(vec_t &data, vector<vec_t> bias, vector<vec_t> &kernels,
		vec_t &out_data) {
	assert(data.size() > 0 && out_data.size() > 0 && kernels.size() > 0);
	assert(data.size() == kernels[0].size());
	assert(out_data.size() == kernels.size());

	float_t *skp, *sdp, *snp, sum;
	sdp = &data[0];
	snp = &out_data[0];
	for (int num = 0; num < kernels.size(); num++) {
		sum = 0;
		skp = &kernels[num][0];
		for (int i = 0; i < data.size(); i++) {
			sum += *(sdp + i) * (*(skp + i));
		}
		sum += bias[num][0];
		*(snp + num) = sum;
	}
}

//caculate the sum(a*x_0i+b)
void CACU_RESET_CPU(vector<vec_t> &data) {

	float_t *skp;
	int length = data[0].size();

	for (int num = 0; num < data.size(); num++) {
		skp = &data[num][0];
		for (int i = 0; i < length; i++) {
			*(skp + i) = 0;
		}
	}
}

#endif

}
;
