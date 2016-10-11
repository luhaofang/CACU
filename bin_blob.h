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

#include "./core/math.hpp"

#if GPU_MODE
#include <cuda_runtime.h>
#define CHECK(res) if(res!=cudaSuccess){printf("[cuda error  %d]\n",res);exit(-1);}
#endif

namespace mycnn {

class bin_blob {

public:

#if GPU_MODE

	bin_blob(int num,int channel,int dim,type phrase = test) {

		this->num = num;
		this->channel = channel;
		this->dim = dim;
		this->phrase = phrase;
		cudaError_t res;
		size_t width = channel * dim * dim;
		vec_i a(num*width);

		unsigned int *d_data;
		unsigned int **h_data = (unsigned int **)malloc(num* sizeof(unsigned int*));

		res = cudaMalloc((void**) (&bin_data), num * sizeof(unsigned int*));
		CHECK(res);
		res = cudaMalloc((void**) (&d_data),num * width * sizeof(unsigned int));
		CHECK(res);

		res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
				num*width * sizeof(unsigned int), cudaMemcpyHostToDevice);

		s_bin_data = d_data;

		for(int i =0; i <num; i++)
		{
			h_data[i] = d_data + i * width;
		}
		res = cudaMemcpy((void*) (bin_data), (void*) (h_data),
				num * sizeof(unsigned int*), cudaMemcpyHostToDevice);
		CHECK(res);

		vec_i().swap(a);

		if (train == phrase) {
			float_t *d_diff;
			float_t **h_diff = (float_t **)malloc(num* sizeof(float_t*));

			vec_t b(num*width);

			res = cudaMalloc((void**) (&diff), num * sizeof(float_t*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_diff),num * width * sizeof(float_t));
			CHECK(res);

			res = cudaMemcpy((void*) (d_diff), (void*) (&b[0]),
					num*width * sizeof(float_t), cudaMemcpyHostToDevice);

			s_diff = d_diff;

			for(int i =0; i <num; i++)
			{
				h_diff[i] = d_diff + i * width;
			}
			res = cudaMemcpy((void*) (diff), (void*) (h_diff),
					num * sizeof(float_t*), cudaMemcpyHostToDevice);
			CHECK(res);
			vec_t().swap(b);
		}
	}

	bin_blob(int num, int length, type phrase = test) {

		this->num = num;
		this->channel = 1;
		this->dim = length;
		this->phrase = phrase;
		cudaError_t res;

		unsigned int *d_data;
		unsigned int **h_data = (unsigned int **)malloc(num* sizeof(unsigned int*));

		vec_i a(num*length);

		res = cudaMalloc((void**) (&bin_data), num * sizeof(unsigned int*));
		CHECK(res);
		res = cudaMalloc((void**) (&d_data),num * length * sizeof(unsigned int));
		CHECK(res);

		res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
				num*length * sizeof(unsigned int), cudaMemcpyHostToDevice);

		s_bin_data = d_data;

		for(int i =0; i <num; i++)
		{
			h_data[i] = d_data + i * length;
		}
		res = cudaMemcpy((void*) (bin_data), (void*) (h_data),
				num * sizeof(unsigned int*), cudaMemcpyHostToDevice);
		CHECK(res);

		vec_i().swap(a);

		if (train == phrase) {
			float_t *d_diff;
			float_t **h_diff = (float_t **)malloc(num* sizeof(float_t*));

			vec_t b(num*length);

			res = cudaMalloc((void**) (&diff), num * sizeof(float_t*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_diff),num * length * sizeof(float_t));
			CHECK(res);

			res = cudaMemcpy((void*) (d_diff), (void*) (&b[0]),
					num*length * sizeof(float_t), cudaMemcpyHostToDevice);

			s_diff = d_diff;

			for(int i =0; i <num; i++)
			{
				h_diff[i] = d_diff + i * length;
			}
			res = cudaMemcpy((void*) (diff), (void*) (h_diff),
					num * sizeof(float_t*), cudaMemcpyHostToDevice);
			CHECK(res);
			vec_t().swap(b);
		}
	}

	~bin_blob() {
		if(train == this->phrase) {
			cudaFree(s_bin_data);
			cudaFree(s_diff);
			cudaFree(bin_data);
			cudaFree(diff);
			s_bin_data = NULL;
			s_diff = NULL;
			for(int i = 0; i < num; i ++)
			{
				bin_data[i] = NULL;
				diff[i] = NULL;
			}
		}
		else {
			cudaFree(s_bin_data);
			cudaFree(bin_data);
			s_bin_data = NULL;
			for(int i = 0; i < num; i++)
			{
				bin_data[i] = NULL;
			}
		}
	};

	void _RESET_DATA()
	{

		if(NULL != bin_data) {
			size_t width = channel * dim * dim;

			CACU_RESET_BIN_DATA_GPU(bin_data,num,width);

			if (train == phrase) {
				CACU_RESET_DATA_GPU(diff,num,width);
			}
		}
	}

	unsigned int **bin_data = NULL;

	float_t **diff = NULL;

	unsigned int *s_bin_data = NULL;

	float_t *s_diff = NULL;

#else
	bin_blob(int num, int channel, int dim, type phrase = test) {

		this->num = num;
		this->channel = channel;
		this->dim = dim;
		this->phrase = phrase;
		this->bin_data = vector<dynamic_bitset<>>(num,
				dynamic_bitset<>(channel * dim * dim));
		if (train == phrase)
			this->diff = vector<vec_t>(num, vec_t(channel * dim * dim));

	}

	bin_blob(int num, int length, type phrase = test) {

		this->num = num;
		this->channel = 1;
		this->dim = length;
		this->phrase = phrase;
		this->bin_data = vector<dynamic_bitset<>>(num,
				dynamic_bitset<>(length));
		if (train == phrase)
			this->diff = vector<vec_t>(num, vec_t(length));

	}

	bin_blob(const bin_blob &copy) {

		this->phrase = copy.phrase;

		this->bin_data = copy.bin_data;

		this->diff = copy.diff;

		this->num = num;

		this->channel = channel;

		this->dim = dim;

	}

	bin_blob(type phrase) {
		this->phrase = phrase;
		this->num = 0;

		this->channel = 0;

		this->dim = 0;
	}

	bin_blob() {
		this->phrase = test;
		this->num = 0;

		this->channel = 0;

		this->dim = 0;
	}

	~bin_blob() {
		for (int i = 0; i < bin_data.size(); i++) {
			dynamic_bitset<>().swap(bin_data[i]);
		}
		for (int i = 0; i < diff.size(); i++) {
			vec_t().swap(diff[i]);
		}
		vector<dynamic_bitset<>>().swap(bin_data);
		vector<vec_t>().swap(diff);
	}
	;

	void _RESET_DATA() {

		for (int n = 0; n < num; n++) {
			bin_data[n].reset();
			if (train == phrase)
				diff[n].resize(channel * dim * dim, 0.0);
		}

	}

	vector<dynamic_bitset<>> bin_data;

	vector<vec_t> diff;

#endif

	int num;

	int channel;

	int dim;

private:
	type phrase;
};

class bin_blobs: public vector<bin_blob*> {

public:

	bin_blobs() {

	}

	~bin_blobs() {
		if (pbin_data.size() > 0)
			for (int i = 0; i < pbin_data.size(); i++) {
				pbin_data[i] = NULL;
			}
		vector<bin_blob**>().swap(pbin_data);
	}

	bin_blobs* operator <<(bin_blob *&_bin_blob) {

		this->push_back(_bin_blob);
		this->pbin_data.push_back(&_bin_blob);

		return this;
	}

	vector<bin_blob**> pbin_data;

private:

};
}
;
