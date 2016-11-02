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

#endif

namespace mycnn {

class blob {

public:

#if GPU_MODE

	blob(int num, int channel, int dim, type phrase =
			test) {

		cudaError_t res;
		size_t width = channel * dim * dim;
		vec_t a(num*width,0);
		float_t *d_data;
		float_t **h_data = (float_t **)malloc(num* sizeof(float_t*));

		res = cudaMalloc((void**) (&data), num * sizeof(float_t*));
		CHECK(res);
		res = cudaMalloc((void**) (&d_data),num * width * sizeof(float_t));
		CHECK(res);

		res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
				num * width * sizeof(float_t), cudaMemcpyHostToDevice);

		s_data = d_data;

		for(int i =0; i <num; i++)
		{
			h_data[i] = d_data + i * width;
		}
		res = cudaMemcpy((void*) (data), (void*) (h_data),
				num * sizeof(float_t*), cudaMemcpyHostToDevice);
		CHECK(res);

		if (train == phrase) {
			float_t *d_diff;
			float_t **h_diff = (float_t **)malloc(num* sizeof(float_t*));

			res = cudaMalloc((void**) (&diff), num * sizeof(float_t*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_diff),num * width * sizeof(float_t));
			CHECK(res);

			res = cudaMemcpy((void*) (d_diff), (void*) (&a[0]),
					num * width * sizeof(float_t), cudaMemcpyHostToDevice);

			s_diff = d_diff;

			for(int i =0; i <num; i++)
			{
				h_diff[i] = d_diff + i * width;
			}
			res = cudaMemcpy((void*) (diff), (void*) (h_diff),
					num * sizeof(float_t*), cudaMemcpyHostToDevice);
			CHECK(res);

		}

		this->phrase = phrase;

		this->num = num;

		this->channel = channel;

		this->dim = dim;
	}

	blob(int num, int length, type phrase = test) {

		this->phrase = phrase;

		cudaError_t res;
		vec_t a(num*length,0);
		float_t *d_data;
		float_t **h_data = (float_t **)malloc(num* sizeof(float_t*));

		res = cudaMalloc((void**) (&data), num * sizeof(float_t*));
		CHECK(res);
		res = cudaMalloc((void**) (&d_data),num * length * sizeof(float_t));
		CHECK(res);

		res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
				num * length * sizeof(float_t), cudaMemcpyHostToDevice);

		s_data = d_data;

		for(int i =0; i <num; i++)
		{
			h_data[i] = d_data + i * length;
		}
		res = cudaMemcpy((void*) (data), (void*) (h_data),
				num * sizeof(float_t*), cudaMemcpyHostToDevice);
		CHECK(res);

		if (train == phrase) {
			float_t *d_diff;
			float_t **h_diff = (float_t **)malloc(num* sizeof(float_t*));

			res = cudaMalloc((void**) (&diff), num * sizeof(float_t*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_diff),num * length * sizeof(float_t));
			CHECK(res);

			res = cudaMemcpy((void*) (d_diff), (void*) (&a[0]),
					num * length * sizeof(float_t), cudaMemcpyHostToDevice);

			s_diff = d_diff;

			for(int i =0; i <num; i++)
			{
				h_diff[i] = d_diff + i * length;
			}
			res = cudaMemcpy((void*) (diff), (void*) (h_diff),
					num * sizeof(float_t*), cudaMemcpyHostToDevice);
			CHECK(res);
		}

		this->num = num;

		this->channel = 1;

		this->dim = length;

		vec_t().swap(a);
	}

	blob(int num, int channel, int dim, float_t value, type phrase = test) {

		cudaError_t res;
		size_t width = channel * dim * dim;
		vec_t a(num*width,value);

		float_t *d_data;
		float_t **h_data = (float_t **)malloc(num* sizeof(float_t*));

		res = cudaMalloc((void**) (&d_data),num * width * sizeof(float_t));
		CHECK(res);

		res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
				num*width * sizeof(float_t), cudaMemcpyHostToDevice);

		s_data = d_data;

		for(int i =0; i <num; i++)
		{
			h_data[i] = d_data + i * width;
		}

		res = cudaMalloc((void**) (&data), num * sizeof(float_t*));
		CHECK(res);
		res = cudaMemcpy((void*) (data), (void*) (h_data),
				num * sizeof(float_t*), cudaMemcpyHostToDevice);
		CHECK(res);

		if (train == phrase) {
			float_t *d_diff;
			float_t **h_diff = (float_t **)malloc(num* sizeof(float_t*));

			res = cudaMalloc((void**) (&d_diff),num * width * sizeof(float_t));
			CHECK(res);

			res = cudaMemcpy((void*) (d_diff), (void*) (&a[0]),
					num*width * sizeof(float_t), cudaMemcpyHostToDevice);

			s_diff = d_diff;

			for(int i =0; i <num; i++)
			{
				h_diff[i] = d_diff + i * width;
			}
			res = cudaMalloc((void**) (&diff), num * sizeof(float_t*));
			CHECK(res);
			res = cudaMemcpy((void*) (diff), (void*) (h_diff),
					num * sizeof(float_t*), cudaMemcpyHostToDevice);
			CHECK(res);
		}

		this->phrase = phrase;

		this->num = num;

		this->channel = channel;

		this->dim = dim;

		vec_t().swap(a);
	}

	~blob() {

		if(train == this->phrase) {
			cudaFree(s_data);
			cudaFree(s_diff);
			//cudaFree(data);
			//cudaFree(diff);
//			s_data = NULL;
//			s_diff = NULL;
//			for(int i = 0; i < num; i ++)
//			{
//				data[i] = NULL;
//				diff[i] = NULL;
//			}
		}
		else {
			cudaFree(s_data);
			//cudaFree(data);
//			s_data = NULL;
//			for(int i = 0; i < num; i++)
//			{
//				data[i] = NULL;
//				cout << "finished" << std::endl;
//			}
		}
	}

	void _RESET_DATA() {

		if(NULL != data) {

			size_t width = channel * dim * dim;

			CACU_RESET_DATA_GPU(data,num,width);

			if (train == phrase) {
				CACU_RESET_DATA_GPU(diff,num,width);
			}
		}
	}

	float_t **data = NULL;

	float_t **diff = NULL;

	float_t *s_data = NULL;

	float_t *s_diff = NULL;

#else

	blob(int num, int channel, int dim, type phrase = test) {

		this->phrase = phrase;
		data = vector<vec_t>(num, vec_t(channel * dim * dim));
		if (train == phrase)
			diff = vector<vec_t>(num, vec_t(channel * dim * dim));

		this->num = num;

		this->channel = channel;

		this->dim = dim;
	}

	blob(int num, int length, type phrase = test) {

		this->phrase = phrase;
		data = vector<vec_t>(num, vec_t(length));
		if (train == phrase)
			diff = vector<vec_t>(num, vec_t(length));

		this->num = num;

		this->channel = 1;

		this->dim = length;
	}

	blob(const blob &copy) {

		this->phrase = copy.phrase;

		this->data = copy.data;

		this->diff = copy.diff;

		this->num = num;

		this->channel = channel;

		this->dim = dim;

	}

	blob(int num, int channel, int dim,float_t value, type phrase = test) {

		this->phrase = phrase;
		data = vector<vec_t>(num, vec_t(channel * dim * dim,value));
		if (train == phrase)
			diff = vector<vec_t>(num, vec_t(channel * dim * dim));

		this->num = num;

		this->channel = channel;

		this->dim = dim;
	}

	blob(type phrase) {
		this->phrase = phrase;
		this->num = 0;

		this->channel = 0;

		this->dim = 0;
	}

	blob() {
		this->phrase = test;
		this->num = 0;

		this->channel = 0;

		this->dim = 0;
	}

	~blob() {

		for (int i = 0; i < this->data.size(); i++) {
			vec_t().swap(data[i]);
		}
		for (int i = 0; i < this->diff.size(); i++) {
			vec_t().swap(diff[i]);
		}
		vector<vec_t>().swap(data);
		vector<vec_t>().swap(diff);
	}

	void _RESET_DATA() {
		for (int n = 0; n < num; n++) {
			for (int i = 0; i < data[0].size(); i++)
				data[n][i] = 0.0;
			if (train == phrase) {
				for (int i = 0; i < diff[0].size(); i++)
					diff[n][i] = 0.0;
			}
		}
	}

	vector<vec_t> data;

	vector<vec_t> diff;

#endif

	int num;

	int channel;

	int dim;

private:

	type phrase;

};

class blobs: public vector<blob*> {

public:

	blobs() {

	}

	~blobs() {

		if (pdata.size() > 0)
			for (int i = 0; i < pdata.size(); i++) {
				pdata[i] = NULL;
			}
		vector<blob**>().swap(pdata);
	}

	blobs* operator <<(blob *&_blob) {

		this->push_back(_blob);
		this->pdata.push_back(&_blob);
		return this;
	}

	vector<blob**> pdata;

private:

};

}
;
