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

#include "../blob.h"
#include "../bin_blob.h"

namespace mycnn {

#if GPU_MODE

extern "C" void copy_padding_data_blob_gpu(float_t *&data, int num,
		int input_dim, int channel, int pad,
		float_t *&out_data);

extern "C" void append_padding_data_blob_gpu(float_t **&data, int num,
		int input_dim, int channel, int pad,
		float_t **&out_data);

extern "C" void copy_unpadding_data_gpu(float_t *&data, int num, int input_dim,
		int channel, int pad, float_t *&out_data);

extern "C" void append_unpadding_data_gpu(float_t **&data, int num,
		int input_dim, int channel, int pad, float_t **&out_data);

extern "C" void copy_padding_data_sign_gpu(unsigned int *&data, int num,
		int input_dim, int channel, int pad,
		unsigned int *&out_data);

extern "C" void img2col_gpu(float_t *&data, int num, int channel,
		int input_dim, int kernel_size, int stride, int output_dim,
		float_t *&pad_input);

extern "C" void col2img_gpu(float_t *&data, int num, int channel,
		int input_dim, int kernel_size, int stride, int output_dim,
		float_t *&pad_input);

//pad and to bitcol
extern "C" void img2bitcol_gpu(unsigned int *&bin_data, int num, int channel,
		int input_dim, int kernel_size, int stride, int pad, int output_dim,
		unsigned int *&pad_input);

extern "C" void copy_data_gpu(float_t **&data, float_t **&out_data,
		int num, int length, int add);

extern "C" void copy_data_bin_gpu(unsigned int **&data,
		unsigned int **&out_data, int num, int length, int add);

extern "C" void copy2dest_gpu(float_t **&data, float_t **&index_data, int num,
		int output_dim, int input_dim, int channel, int kernel_size, int stride,
		float_t **&out_data);

extern "C" void copy2mean_gpu(float_t **&data, int num,int output_dim, int input_dim,
		int channel, int kernel_size, int stride, int pad, float_t **&out_data);

extern "C" void reset_data_gpu(float_t *&data, int num, int length);

extern "C" void reset_bin_data_gpu(unsigned int *&data, int num, int length);

extern "C" void set_data_gpu(float_t **&data, int num, int length,
		float_t value);

#else

	void copy_padding_data_blob(vector<vec_t> &data, int input_dim, int pad, vector<vec_t> &signdata) {


		int output_dim_ = input_dim + (2 * pad);
		int channel_ = data[0].size() / (input_dim * input_dim);
		int length = output_dim_ * output_dim_ * channel_;
		int flag = pad + input_dim;
		assert(signdata.size() == data.size());
		assert(signdata[0].size() == length);

		for (int num = 0; num < data.size(); num++) {

			float_t *s_fp = &signdata[num][0];
			float_t *sp = &data[num][0];

			float_t *p, *dp;

			for (int i = 0; i < output_dim_; i++) {
				for (int j = 0; j < output_dim_; j++) {
					if ((i >= pad && i < flag) && (j >= pad && j < flag)) {
						for (int c = 0; c < channel_; c++) {
							p = s_fp + (i * output_dim_ + j) * channel_ + c;
							dp = sp + ((i - pad) * input_dim + (j - pad)) * channel_
								+ c;
							*p = *dp;
						}
					}
				}
			}
		}
	}

	void append_padding_data_blob(vector<vec_t> &data, int input_dim, int pad, vector<vec_t> &signdata) {
	
	int output_dim_ = input_dim + pad;
	int channel_ = data[0].size() / (input_dim * input_dim);
	int length = output_dim_ * output_dim_ * channel_;
	assert(signdata.size() == data.size());
	assert(signdata[0].size() == length);

	for (int num = 0; num < data.size(); num++) {

		float_t *s_fp = &signdata[num][0];
		float_t *sp = &data[num][0];

		float_t *p, *dp;

		for (int i = 0; i < output_dim_; i++) {
			for (int j = 0; j < output_dim_; j++) {
				if ((i < input_dim) && (j < input_dim)) {
					for (int c = 0; c < channel_; c++) {
						p = s_fp + (i * output_dim_ + j) * channel_ + c;
						dp = sp + (i * input_dim + j) * channel_ + c;
						*p = *dp;
					}
				}
			}
		}
	}
}

void copy_unpadding_data(vector<vec_t> &data, int input_dim, int pad,
		vector<vec_t> &out_data) {
	assert(data.size() == out_data.size());
	int input_dim_ = input_dim + (2 * pad);
	int channel_ = data[0].size() / (input_dim_ * input_dim_);
	float_t *p, *np;
	for (int num = 0; num < data.size(); num++) {
		float_t *sp = &data[num][0];
		float_t *sn = &out_data[num][0];
		for (int i = 0; i < input_dim_; i++) {
			for (int j = 0; j < input_dim_; j++) {
				if (i >= pad && i < pad + input_dim && j >= pad
						&& j < pad + input_dim) {
					p = sp + (i * input_dim_ + j) * channel_;
					np = sn + ((i - pad) * (input_dim) + j - pad) * channel_;
					for (int c = 0; c < channel_; c++)
						*(np + c) = (*(p + c));
				}
			}
		}
	}
}

void append_unpadding_data(vector<vec_t> &data, int input_dim, int pad,
		vector<vec_t> &out_data) {
	assert(data.size() == out_data.size());
	int input_dim_ = input_dim + pad;
	int channel_ = data[0].size() / (input_dim_ * input_dim_);
	assert(out_data[0].size() == channel_ * input_dim * input_dim);
	float_t *p, *np;
	for (int num = 0; num < data.size(); num++) {
		vec_t sign_i;
		float_t *sp = &data[num][0];
		float_t *sn = &out_data[num][0];
		for (int i = 0; i < input_dim; i++) {
			for (int j = 0; j < input_dim; j++) {
				if (i < input_dim && j < input_dim) {
					p = sp + (i * input_dim_ + j) * channel_;
					np = sn + (i * input_dim + j) * channel_;
					for (int c = 0; c < channel_; c++)
						*(np + c) = (*(p + c));
				}
			}
		}
	}
}

void copy_padding_data_sign(vector<dynamic_bitset<>> &data, int input_dim,
	int pad, vector<dynamic_bitset<>> &signdata) {
	
	int output_dim_ = input_dim + (2 * pad);
	int channel_ = data[0].size() / (input_dim * input_dim);
	int length = output_dim_ * output_dim_ * channel_;
	int flag = pad + input_dim;

	for (int num = 0; num < data.size(); num++) {
		dynamic_bitset<> sign_i;
		for (int i = 0; i < output_dim_; i++) {
			for (int j = 0; j < output_dim_; j++) {
				if (i < pad || i >= flag)
					for (int c = 0; c < channel_; c++)
						sign_i.push_back(0);
				else {
					if (j < pad || j >= flag)
						for (int c = 0; c < channel_; c++)
							sign_i.push_back(0);
					else {
						int p = ((i - pad) * input_dim + (j - pad)) * channel_;
						for (int c = 0; c < channel_; c++)
							sign_i.push_back(data[num][p + c]);
					}
				}
			}
		}
		signdata.push_back(sign_i);
	}
}

void img2col(vector<vec_t> &data, int kernel_size, int stride, int pad,
		int input_dim, int output_dim, vector<vec_t> &out_data) {
	assert(data.size() > 0);
	int channel = (data[0].size() / (input_dim * input_dim));
	assert(
			out_data[0].size()
					== kernel_size * kernel_size * channel * output_dim
							* output_dim);

	float_t *sdp, *snp;
	int sd_out, sn_out;
	float_t *sd_out_cp, *sn_out_cp;

	blob *_data = new blob(data.size(), channel*(input_dim + 2 * pad)*(input_dim + 2 * pad));
	copy_padding_data_blob(data, input_dim, pad, _data->data);

	input_dim = input_dim + 2 * pad;

	for (int num = 0; num < _data->data.size(); num++) {

		sdp = &_data->data[num][0];
		snp = &out_data[num][0];

		//for output_dim's iteration
		for (int i = 0; i < output_dim; i++)
			for (int j = 0; j < output_dim; j++) {
				sd_out = (i * input_dim + j) * stride * channel;
				sn_out = (i * output_dim + j) * channel;
				//for channel's iteration
				for (int c = 0; c < channel; c++) {
					sd_out_cp = sdp + sd_out + c;
					sn_out_cp = snp + (sn_out + c) * kernel_size * kernel_size;
					//for kernel_size 's iteration
					for (int ki = 0; ki < kernel_size; ki++)
						for (int kj = 0; kj < kernel_size; kj++) {
							*(sn_out_cp + ki * kernel_size + kj) = *(sd_out_cp
									+ (ki * input_dim + kj) * channel);
						}
				}
			}
	}
	delete _data;
}

void img2bitcol(vector<dynamic_bitset<>> &data, int channel,int kernel_size, int stride,
		int pad, int input_dim, int output_dim, vector<vec_i> &out_data) {
	assert(data.size() > 0);
	int length;
	if (channel * kernel_size * kernel_size % BIN_SIZE == 0)
		length = (channel * kernel_size * kernel_size / BIN_SIZE) * output_dim
				* output_dim;
	else
		length = (channel * kernel_size * kernel_size / BIN_SIZE + 1)
				* output_dim * output_dim;

	assert(length == out_data[0].size());

	unsigned int *snp;
	dynamic_bitset<> sbp(BIN_SIZE);
	dynamic_bitset<> sdp;
	int sd_out, sn_out;
	int sd_out_c;

	bin_blob *_data = new bin_blob();
	copy_padding_data_sign(data, input_dim, pad, _data->bin_data);

	input_dim = input_dim + 2 * pad;

	int count = 0, motif = kernel_size * kernel_size * channel, index;
	int flag = BIN_SIZE - 1, end_flag = motif - 1;

	for (int num = 0; num < _data->bin_data.size(); num++) {

		sdp = _data->bin_data[num];
		snp = &out_data[num][0];
		//for output_dim's iteration
		for (int i = 0; i < output_dim; i++)
			for (int j = 0; j < output_dim; j++) {
				sd_out = (i * input_dim + j) * stride;
				sn_out = i * output_dim + j;
				//to unlong
				count = 0;
				//for channel's iteration
				for (int c = 0; c < channel; c++) {
					sd_out_c = sd_out * channel + c;
					//for kernel_size 's iteration
					for (int ki = 0; ki < kernel_size; ki++)
						for (int kj = 0; kj < kernel_size; kj++) {
							index = count % BIN_SIZE;
							sbp[index] = sdp[sd_out_c
									+ (ki * input_dim + kj) * channel];
							if (index == flag || count == end_flag) {
								*snp = sbp.to_ulong();
								sbp.reset();
								snp += 1;
							}
							count++;
						}
				}
			}
	}
	delete _data;
}

void img2col(vector<vec_t> &data, int kernel_size, int stride, int pad,
		int input_dim, int output_dim, vector<vector<vec_t>> &out_data) {
	assert(data.size() > 0);
	int channel = (data[0].size() / (input_dim * input_dim));
	assert(out_data[0][0].size() == kernel_size * kernel_size * channel);

	float_t *sdp, *snp;
	int sd_out;
	float_t *sd_out_cp, *sn_out_cp;

	blob *_data = new blob(data.size(), channel*(input_dim + 2 * pad)*(input_dim + 2 * pad));
	copy_padding_data_blob(data, input_dim, pad, _data->data);

	input_dim = input_dim + 2 * pad;

	for (int num = 0; num < _data->data.size(); num++) {

		sdp = &_data->data[num][0];

		//for output_dim's iteration
		for (int i = 0; i < output_dim; i++)
			for (int j = 0; j < output_dim; j++) {
				snp = &out_data[num][i * output_dim + j][0];
				sd_out = (i * input_dim + j) * stride;
				//for channel's iteration
				for (int c = 0; c < channel; c++) {
					sd_out_cp = sdp + sd_out * channel + c;
					sn_out_cp = snp + c * kernel_size * kernel_size;
					//for kernel_size 's iteration
					for (int ki = 0; ki < kernel_size; ki++)
						for (int kj = 0; kj < kernel_size; kj++) {
							*(sn_out_cp + ki * kernel_size + kj) = *(sd_out_cp
									+ (ki * input_dim + kj) * channel);
						}
				}
			}
	}
	delete _data;
}

//caculate the de_convolution for diff
//data : col_data
//out_data : img_data(padded)
void col2img(vector<vec_t> &data, int kernel_size, int stride, int pad,
		int input_dim, int output_dim, vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	input_dim = input_dim + pad * 2;
	int channel = (out_data[0].size() / (input_dim * input_dim));
	assert(
			data[0].size()
					== kernel_size * kernel_size * channel * output_dim
							* output_dim);

	float_t *sdp, *snp;
	int sd_out, sn_out;

	int block_size = kernel_size * kernel_size * channel;
	int k_size = kernel_size * kernel_size;
	int length = data[0].size();
	int out_dim = output_dim * output_dim;
	int border = input_dim - output_dim;

	for (int num = 0; num < data.size(); num++) {

		sdp = &data[num][0];
		snp = &out_data[num][0];

		//for output_dim's location
		for (int index = 0; index < out_dim; index++) {
			sd_out = index * block_size;
			sn_out = ((index / output_dim) * input_dim + (index % output_dim))
					* stride * channel;
			for (int ki = 0; ki < kernel_size; ki++)
				for (int kj = 0; kj < kernel_size; kj++) {
					for (int c = 0; c < channel; c++) {
						*(snp + sn_out + (ki * input_dim + kj) * channel + c) +=
								*(sdp + sd_out + c * k_size + ki * kernel_size
										+ kj);
					}
				}
		}
	}
}

void transpose(vector<vec_t> &data, vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data[0].size());
	assert(out_data.size() == data[0].size());

	int width = data[0].size();
	int height = out_data[0].size();

	float_t *sdp;
	for (int i = 0; i < height; i++) {
		sdp = &data[i][0];
		for (int j = 0; j < width; j++) {
			*(&out_data[j][0] + i) = *(sdp + j);
		}
	}
}

void copy_data(vector<vec_t> &data, vector<vec_t> &out_data, int add) {
	assert(data.size() == out_data.size());
	assert(data[0].size() == out_data[0].size());
	float_t *sdp, *snp;
	int length = data[0].size();
	if (!add)
		for (int num = 0; num < data.size(); num++) {
			sdp = &data[num][0];
			snp = &out_data[num][0];
			for (int i = 0; i < length; i++) {
				*(snp + i) = *(sdp + i);
			}
		}
	else
		for (int num = 0; num < data.size(); num++) {
			sdp = &data[num][0];
			snp = &out_data[num][0];
			for (int i = 0; i < length; i++) {
				*(snp + i) += *(sdp + i);
			}
		}
}

//copy data to directed set
void copy2dest(vector<vec_t> &data, vector<vec_t> &index_data, int output_dim,
		int input_dim, int channel, int kernel_size, int stride,
		vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0 && index_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() == index_data[0].size());
	assert(data[0].size() == output_dim * output_dim * channel);

	float_t *sdp, *snp;
	int sd_out, si_out, index;
	float_t *sd_out_cp, *sn_out_cp;
	float_t *sip, *si_out_cp;

	int xi, xj;

	for (int num = 0; num < data.size(); num++) {
		sdp = &data[num][0];
		sip = &index_data[num][0];
		snp = &out_data[num][0];

		for (int i = 0; i < output_dim; i++)
			for (int j = 0; j < output_dim; j++) {
				sd_out = (i * output_dim + j) * channel;
				si_out = sd_out;
				for (int c = 0; c < channel; c++) {
					si_out_cp = sip + si_out + c;
					sd_out_cp = sdp + sd_out + c;
					index = (int) *si_out_cp;
					xi = (i * stride + index / kernel_size);
					xj = (j * stride + index % kernel_size);
					sn_out_cp = snp + ((xi * input_dim + xj) * channel + c);
					*sn_out_cp += *sd_out_cp;
				}
			}
	}
}

void copy2mean(vector<vec_t> &data, int output_dim, int input_dim, int channel,
		int kernel_size, int stride, int pad, vector<vec_t> &out_data) {
	assert(data.size() > 0 && out_data.size() > 0);
	assert(data.size() == out_data.size());
	assert(data[0].size() == output_dim * output_dim * channel);

	float_t *sdp, *snp;
	int sd_out, sn_out, param_w, param_h;
	float_t *sd_out_cp, *sn_out_cp;
	float_t diff_data;
	int flag = output_dim - 1;
	int input_dim_ = input_dim + pad;

	for (int num = 0; num < data.size(); num++) {
		sdp = &data[num][0];
		snp = &out_data[num][0];

		for (int i = 0; i < output_dim; i++)
			for (int j = 0; j < output_dim; j++) {
				sd_out = (i * output_dim + j) * channel;
				sn_out = (i * input_dim_ + j) * stride * channel;
				for (int c = 0; c < channel; c++) {
					sd_out_cp = sdp + sd_out + c;
					//mean
					if (pad == 0)
						diff_data = *sd_out_cp
								/ (float_t) (kernel_size * kernel_size);
					else {
						param_w = kernel_size, param_h = kernel_size;
						if (i == flag)
							param_w = kernel_size - pad;

						if (j == flag)
							param_h = kernel_size - pad;
						diff_data = *sd_out_cp / (float_t) (param_w * param_h);
					}
					for (int ki = 0; ki < kernel_size; ki++)
						for (int kj = 0; kj < kernel_size; kj++) {
							sn_out_cp = snp + sn_out
									+ (ki * input_dim_ + kj) * channel + c;
							*sn_out_cp += diff_data;
						}
				}
			}
	}
}

#endif

}
;
