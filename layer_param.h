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

#include "core/matrix.hpp"

#if GPU_MODE
#include <cuda_runtime.h>
#endif

namespace mycnn {

class layer_param {

public:

#if GPU_MODE

	layer_param(map<char_t, int> _param_outnum,
			map<char_t, int> _param_dim,
			map<char_t, int> _bin_param_outnum,
			map<char_t, int> _bin_param_dim)
	{
		cudaError_t res;
		map<char_t, int>::iterator it;
		for (it = _param_outnum.begin(); it != _param_outnum.end(); ++it)
		{
			float_t *d_data;
			vec_t a(it->second * _param_dim[it->first],0);
			float_t **h_data = (float_t **)malloc(it->second* sizeof(float_t*));

			res = cudaMalloc((void**) (&data[it->first]), it->second * sizeof(float_t*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_data),it->second * _param_dim[it->first] * sizeof(float_t));
			CHECK(res);

			res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
					it->second * _param_dim[it->first] * sizeof(float_t), cudaMemcpyHostToDevice);

			s_data[it->first] = d_data;

			for(int i =0; i < it->second; i++)
			{
				h_data[i] = d_data + i * _param_dim[it->first];
			}
			res = cudaMemcpy((void*) (data[it->first]), (void*) (h_data),
					it->second * sizeof(float_t*), cudaMemcpyHostToDevice);
			CHECK(res);

			vec_t().swap(a);
		}
		map<char_t, int>::iterator bin_it;
		for (bin_it = _bin_param_outnum.begin(); bin_it != _bin_param_outnum.end(); ++bin_it)
		{
			bin_data[bin_it->first] = (unsigned int **) malloc(bin_it->second * sizeof(unsigned int*));
			int length;
			if (_bin_param_dim[bin_it->first] % 32 == 0)
			length = (_bin_param_dim[bin_it->first] / 32);
			else
			length = (_bin_param_dim[bin_it->first] / 32 + 1);

			unsigned int *d_data;
			unsigned int **h_data = (unsigned int **)malloc(bin_it->second* sizeof(unsigned int*));

			vec_i b(bin_it->second * length,0);

			res = cudaMalloc((void**) (&bin_data[bin_it->first]), bin_it->second * sizeof(unsigned int*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_data),bin_it->second * length * sizeof(unsigned int));
			CHECK(res);

			res = cudaMemcpy((void*) (d_data), (void*) (&b[0]),
					bin_it->second * length * sizeof(unsigned int), cudaMemcpyHostToDevice);

			s_bin_data[bin_it->first] = d_data;

			for(int i =0; i < bin_it->second; i++)
			{
				h_data[i] = d_data + i * length;
			}
			res = cudaMemcpy((void*) (bin_data[bin_it->first]), (void*) (h_data),
					bin_it->second * sizeof(unsigned int*), cudaMemcpyHostToDevice);
			CHECK(res);

			vec_i().swap(b);
		}

		this->param_dim = _param_dim;
		this->param_outnum = _param_outnum;
		this->bin_param_dim = _bin_param_dim;
		this->bin_param_outnum = _bin_param_outnum;
	}

	layer_param(pP_SPACE _pPARAMS)
	{
		cudaError_t res;
		map<char_t, int>::iterator it;
		for (it = _pPARAMS[0].begin(); it != _pPARAMS[0].end(); ++it)
		{

			float_t *d_data;
			float_t **h_data = (float_t **)malloc(it->second* sizeof(float_t*));

			vec_t a(it->second * _pPARAMS[1][it->first],0);

			res = cudaMalloc((void**) (&data[it->first]), it->second * sizeof(float_t*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_data),it->second * _pPARAMS[1][it->first] * sizeof(float_t));
			CHECK(res);

			res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
					it->second * _pPARAMS[1][it->first] * sizeof(float_t), cudaMemcpyHostToDevice);

			s_data[it->first] = d_data;

			for(int i =0; i < it->second; i++)
			{
				h_data[i] = d_data + i * _pPARAMS[1][it->first];
			}
			res = cudaMemcpy((void*) (data[it->first]), (void*) (h_data),
					it->second * sizeof(float_t*), cudaMemcpyHostToDevice);
			CHECK(res);
			vec_t().swap(a);
		}

		map<char_t, int>::iterator bin_it;
		for (bin_it = _pPARAMS[2].begin(); bin_it != _pPARAMS[2].end(); ++bin_it)
		{
			bin_data[bin_it->first] = (unsigned int **) malloc(bin_it->second * sizeof(unsigned int*));
			int length;
			if (_pPARAMS[3][bin_it->first] % 32 == 0)
			length = (_pPARAMS[3][bin_it->first] / 32);
			else
			length = (_pPARAMS[3][bin_it->first] / 32 + 1);

			unsigned int *d_data;
			unsigned int **h_data = (unsigned int **)malloc(bin_it->second* sizeof(unsigned int*));
			vec_i b(bin_it->second * length,0);

			res = cudaMalloc((void**) (&bin_data[bin_it->first]), bin_it->second * sizeof(unsigned int*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_data),bin_it->second * length * sizeof(unsigned int));
			CHECK(res);

			res = cudaMemcpy((void*) (d_data), (void*) (&b[0]),
					bin_it->second * length * sizeof(unsigned int), cudaMemcpyHostToDevice);

			s_bin_data[bin_it->first] = d_data;

			for(int i =0; i < bin_it->second; i++)
			{
				h_data[i] = d_data + i * length;
			}
			res = cudaMemcpy((void*) (bin_data[bin_it->first]), (void*) (h_data),
					bin_it->second * sizeof(unsigned int*), cudaMemcpyHostToDevice);
			CHECK(res);

			vec_i().swap(b);
		}

		this->param_dim = _pPARAMS[1];
		this->param_outnum = _pPARAMS[0];
		this->bin_param_dim = _pPARAMS[3];
		this->bin_param_outnum = _pPARAMS[2];
	}

	//initial params
	layer_param(pP_SPACE _pPARAMS, pP_SPACE_INIT_TYPE _pTYPE, pP_SPACE_INIT_VALUE _pVALUE)
	{
		cudaError_t res;
		random *r = new random();
		map<char_t, int>::iterator it;
		for (it = _pPARAMS[0].begin(); it != _pPARAMS[0].end(); ++it)
		{
			param _param;
			if (_pTYPE.find(it->first) != _pTYPE.end())
			{
				switch (_pTYPE[it->first]) {
					case constant:
					{
						vec_t a(it->second * _pPARAMS[1][it->first], _pVALUE[it->first]);

						float_t *d_data;
						float_t **h_data = (float_t **)malloc(it->second* sizeof(float_t*));

						res = cudaMalloc((void**) (&data[it->first]), it->second * sizeof(float_t*));
						CHECK(res);
						res = cudaMalloc((void**) (&d_data),it->second * _pPARAMS[1][it->first] * sizeof(float_t));
						CHECK(res);

						res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
								it->second * _pPARAMS[1][it->first] * sizeof(float_t), cudaMemcpyHostToDevice);

						s_data[it->first] = d_data;

						for(int i =0; i < it->second; i++)
						{
							h_data[i] = d_data + i * _pPARAMS[1][it->first];
						}
						res = cudaMemcpy((void*) (data[it->first]), (void*) (h_data),
								it->second * sizeof(float_t*), cudaMemcpyHostToDevice);
						CHECK(res);

						vec_t().swap(a);
						break;
					}
					case xavier:
					{
						vec_t a;
						for (int len = 0; len < it->second * _pPARAMS[1][it->first]; len++) {
							a.push_back(r->frand(-_pVALUE[it->first], _pVALUE[it->first]));
						}

						float_t *d_data;
						float_t **h_data = (float_t **)malloc(it->second* sizeof(float_t*));

						res = cudaMalloc((void**) (&data[it->first]), it->second * sizeof(float_t*));
						CHECK(res);
						res = cudaMalloc((void**) (&d_data),it->second * _pPARAMS[1][it->first] * sizeof(float_t));
						CHECK(res);

						res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
								it->second * _pPARAMS[1][it->first] * sizeof(float_t), cudaMemcpyHostToDevice);

						s_data[it->first] = d_data;

						for(int i =0; i < it->second; i++)
						{
							h_data[i] = d_data + i * _pPARAMS[1][it->first];
						}
						res = cudaMemcpy((void*) (data[it->first]), (void*) (h_data),
								it->second * sizeof(float_t*), cudaMemcpyHostToDevice);
						CHECK(res);

						vec_t().swap(a);
						break;
					}
					case gaussian:
					{

						vec_t a;
						for (int len = 0; len < it->second * _pPARAMS[1][it->first]; len++) {
							a.push_back(r->gaussrand(_pVALUE[it->first]));
						}

						float_t *d_data;
						float_t **h_data = (float_t **)malloc(it->second* sizeof(float_t*));

						res = cudaMalloc((void**) (&data[it->first]), it->second * sizeof(float_t*));
						CHECK(res);
						res = cudaMalloc((void**) (&d_data),it->second * _pPARAMS[1][it->first] * sizeof(float_t));
						CHECK(res);

						res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
								it->second * _pPARAMS[1][it->first] * sizeof(float_t), cudaMemcpyHostToDevice);

						s_data[it->first] = d_data;

						for(int i =0; i < it->second; i++)
						{
							h_data[i] = d_data + i * _pPARAMS[1][it->first];
						}
						res = cudaMemcpy((void*) (data[it->first]), (void*) (h_data),
								it->second * sizeof(float_t*), cudaMemcpyHostToDevice);
						CHECK(res);

						vec_t().swap(a);
						break;
					}
					case msra:
					{

						vec_t a;
						for (int len = 0; len < it->second * _pPARAMS[1][it->first]; len++) {
							a.push_back(r->gaussrand(_pVALUE[it->first]));
						}

						float_t *d_data;
						float_t **h_data = (float_t **)malloc(it->second* sizeof(float_t*));

						res = cudaMalloc((void**) (&data[it->first]), it->second * sizeof(float_t*));
						CHECK(res);
						res = cudaMalloc((void**) (&d_data),it->second * _pPARAMS[1][it->first] * sizeof(float_t));
						CHECK(res);

						res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
								it->second * _pPARAMS[1][it->first] * sizeof(float_t), cudaMemcpyHostToDevice);

						s_data[it->first] = d_data;

						for(int i =0; i < it->second; i++)
						{
							h_data[i] = d_data + i * _pPARAMS[1][it->first];
						}
						res = cudaMemcpy((void*) (data[it->first]), (void*) (h_data),
								it->second * sizeof(float_t*), cudaMemcpyHostToDevice);
						CHECK(res);

						vec_t().swap(a);
						break;
					}
				}
			}
			else
			{
				vec_t a(it->second * _pPARAMS[1][it->first] ,0);
				float_t *d_data;
				float_t **h_data = (float_t **)malloc(it->second* sizeof(float_t*));

				res = cudaMalloc((void**) (&data[it->first]), it->second * sizeof(float_t*));
				CHECK(res);
				res = cudaMalloc((void**) (&d_data),it->second * _pPARAMS[1][it->first] * sizeof(float_t));
				CHECK(res);

				res = cudaMemcpy((void*) (d_data), (void*) (&a[0]),
						it->second * _pPARAMS[1][it->first] * sizeof(unsigned int), cudaMemcpyHostToDevice);

				s_data[it->first] = d_data;

				for(int i =0; i < it->second; i++)
				{
					h_data[i] = d_data + i * _pPARAMS[1][it->first];
				}
				res = cudaMemcpy((void*) (data[it->first]), (void*) (h_data),
						it->second * sizeof(float_t*), cudaMemcpyHostToDevice);
				CHECK(res);

				vec_t().swap(a);
			}
		}
		map<char_t, int>::iterator bin_it;
		for (bin_it = _pPARAMS[2].begin(); bin_it != _pPARAMS[2].end(); ++bin_it)
		{
			bin_data[bin_it->first] = (unsigned int **) malloc(bin_it->second * sizeof(unsigned int*));
			int length;
			if (_pPARAMS[3][bin_it->first] % 32 == 0)
			length = (_pPARAMS[3][bin_it->first] / 32);
			else
			length = (_pPARAMS[3][bin_it->first] / 32 + 1);

			unsigned int *d_data;
			unsigned int **h_data = (unsigned int **)malloc(bin_it->second* sizeof(unsigned int*));

			vec_i b(bin_it->second * length,0);

			res = cudaMalloc((void**) (&bin_data[bin_it->first]), bin_it->second * sizeof(unsigned int*));
			CHECK(res);
			res = cudaMalloc((void**) (&d_data),bin_it->second * length * sizeof(unsigned int));
			CHECK(res);

			res = cudaMemcpy((void*) (d_data), (void*) (&b[0]),
					bin_it->second * length * sizeof(unsigned int), cudaMemcpyHostToDevice);

			s_bin_data[bin_it->first] = d_data;

			for(int i =0; i < bin_it->second; i++)
			{
				h_data[i] = d_data + i * length;
			}
			res = cudaMemcpy((void*) (bin_data[bin_it->first]), (void*) (h_data),
					bin_it->second * sizeof(unsigned int*), cudaMemcpyHostToDevice);
			CHECK(res);

			vec_i().swap(b);

		}

		this->param_dim = _pPARAMS[1];
		this->param_outnum = _pPARAMS[0];
		this->bin_param_dim = _pPARAMS[3];
		this->bin_param_outnum = _pPARAMS[2];
	}

	layer_param() {}

	~layer_param() {

		map<char_t, float_t**>::iterator it;
		for (it = data.begin(); it != data.end(); ++it) {
			cudaFree(s_data[it->first]);
			cudaFree(it->second);
			//for(int i =0; i < param_outnum[it->first];i++)
			//it->second[i] = NULL;
		}

		map<char_t, unsigned int**>::iterator b_it;
		for (b_it = bin_data.begin(); b_it != bin_data.end(); ++b_it) {
			cudaFree(s_bin_data[b_it->first]);
			cudaFree(b_it->second);
			//for(int i =0; i < bin_param_outnum[b_it->first];i++)
			//b_it->second[i] = NULL;
		}
		map<char_t, unsigned int**>().swap(bin_data);
		map<char_t, float_t**>().swap(data);

		map<char_t, int>().swap(param_outnum);
		map<char_t, int>().swap(param_dim);
		map<char_t, int>().swap(bin_param_outnum);
		map<char_t, int>().swap(bin_param_dim);

	}

	unsigned int caculate_space()
	{
		int sum = 0;
		int length;
		if (NULL != this)
		{
			map<char_t, int>::iterator it;
			for (it = param_outnum.begin(); it != param_outnum.end(); ++it)
			{
				sum += param_dim[it->first];
			}
			map<char_t, int>::iterator bin_it;
			for (bin_it = bin_param_outnum.begin(); bin_it != bin_param_outnum.end(); ++bin_it)
			{
				if (bin_param_dim[bin_it->first] % 32 == 0)
				length = (bin_param_dim[bin_it->first] / 32);
				else
				length = (bin_param_dim[bin_it->first] / 32 + 1);
				sum += length;
			}
		}
		return sum;
	}

	void _RESET_DATA() {
		if (NULL != this) {
			cudaError_t res;

			map<char_t, int>::iterator it;
			for (it = param_outnum.begin(); it != param_outnum.end(); ++it)
			{
				reset_data_gpu(s_data[it->first],param_outnum[it->first],param_dim[it->first]);
			}
			map<char_t, int>::iterator bin_it;
			for (bin_it = bin_param_outnum.begin(); bin_it != bin_param_outnum.end(); ++bin_it)
			{
				int length;
				if (bin_param_dim[bin_it->first] % 32 == 0)
				length = (bin_param_dim[bin_it->first] / 32);
				else
				length = (bin_param_dim[bin_it->first] / 32 + 1);
				reset_bin_data_gpu(s_bin_data[bin_it->first],bin_param_outnum[bin_it->first],length);
			}
		}
	}

//for normal data
	map<char_t, float_t**> data;
//for binary data
	map<char_t, unsigned int**> bin_data;

//for normal data
	map<char_t, float_t*> s_data;
//for binary data
	map<char_t, unsigned int*> s_bin_data;

#else
	layer_param(map<char_t, int> _param_outnum, map<char_t, int> _param_dim,
			map<char_t, int> _bin_param_outnum,
			map<char_t, int> _bin_param_dim) {
		map<char_t, int>::iterator it;
		for (it = _param_outnum.begin(); it != _param_outnum.end(); ++it) {
			param _param;
			for (unsigned int num = 0; num < it->second; num++) {
				_param.push_back(vec_t(_param_dim[it->first]));
			}
			data[it->first] = _param;
		}
		map<char_t, int>::iterator bin_it;
		for (bin_it = _bin_param_outnum.begin();
				bin_it != _bin_param_outnum.end(); ++bin_it) {
			bin_param _bin_param;
			for (int num = 0; num < bin_it->second; num++) {
				if (_bin_param_dim[bin_it->first] % 32 == 0)
					_bin_param.push_back(
							vec_i(_bin_param_dim[bin_it->first] / 32));
				else
					_bin_param.push_back(
							vec_i(_bin_param_dim[bin_it->first] / 32 + 1));
			}
			bin_data[bin_it->first] = _bin_param;
		}

		this->param_dim = _param_dim;
		this->param_outnum = _param_outnum;
		this->bin_param_dim = _bin_param_dim;
		this->bin_param_outnum = _bin_param_outnum;
	}

	layer_param(pP_SPACE _pPARAMS) {
		map<char_t, int>::iterator it;
		for (it = _pPARAMS[0].begin(); it != _pPARAMS[0].end(); ++it) {
			param _param;
			for (int num = 0; num < it->second; num++) {
				_param.push_back(vec_t(_pPARAMS[1][it->first]));
			}
			data[it->first] = _param;
		}
		map<char_t, int>::iterator bin_it;
		for (bin_it = _pPARAMS[2].begin(); bin_it != _pPARAMS[2].end();
				++bin_it) {
			bin_param _bin_param;
			for (int num = 0; num < bin_it->second; num++) {
				if (_pPARAMS[3][bin_it->first] % 32 == 0)
					_bin_param.push_back(
							vec_i(_pPARAMS[3][bin_it->first] / 32));
				else
					_bin_param.push_back(
							vec_i(_pPARAMS[3][bin_it->first] / 32 + 1));
			}
			bin_data[bin_it->first] = _bin_param;
		}

		this->param_dim = _pPARAMS[1];
		this->param_outnum = _pPARAMS[0];
		this->bin_param_dim = _pPARAMS[3];
		this->bin_param_outnum = _pPARAMS[2];
	}

	//initial params
	layer_param(pP_SPACE _pPARAMS, pP_SPACE_INIT_TYPE _pTYPE,
			pP_SPACE_INIT_VALUE _pVALUE) {

		random *r = new random();

		map<char_t, int>::iterator it;
		for (it = _pPARAMS[0].begin(); it != _pPARAMS[0].end(); ++it) {
			param _param;
			if (_pTYPE.find(it->first) != _pTYPE.end()) {
				switch (_pTYPE[it->first]) {
				case constant:
					for (int num = 0; num < it->second; num++) {
						_param.push_back(
								vec_t(_pPARAMS[1][it->first],
										_pVALUE[it->first]));
					}
					break;
				case xavier:
					for (int num = 0; num < it->second; num++) {
						vec_t a;
						for (int len = 0; len < _pPARAMS[1][it->first]; len++) {
							a.push_back(
									r->frand(-_pVALUE[it->first],
											_pVALUE[it->first]));
						}
						_param.push_back(a);
					}
					break;
				case gaussian:
					for (int num = 0; num < it->second; num++) {
						vec_t a;
						for (int len = 0; len < _pPARAMS[1][it->first]; len++) {
							a.push_back(r->gaussrand(_pVALUE[it->first]));
						}
						_param.push_back(a);
					}
					break;
				case msra:
					for (int num = 0; num < it->second; num++) {
						vec_t a;
						for (int len = 0; len < _pPARAMS[1][it->first]; len++) {
							a.push_back(r->gaussrand(_pVALUE[it->first]));
						}
						_param.push_back(a);
					}
					break;
				}
			} else {
				for (int num = 0; num < it->second; num++) {
					_param.push_back(vec_t(_pPARAMS[1][it->first]));
				};
			}
			data[it->first] = _param;
		}
		map<char_t, int>::iterator bin_it;
		for (bin_it = _pPARAMS[2].begin(); bin_it != _pPARAMS[2].end();
				++bin_it) {
			bin_param _bin_param;
			for (int num = 0; num < bin_it->second; num++) {
				if (_pPARAMS[3][bin_it->first] % 32 == 0)
					_bin_param.push_back(
							vec_i(_pPARAMS[3][bin_it->first] / 32));
				else
					_bin_param.push_back(
							vec_i(_pPARAMS[3][bin_it->first] / 32 + 1));
			}
			bin_data[bin_it->first] = _bin_param;
		}

		this->param_dim = _pPARAMS[1];
		this->param_outnum = _pPARAMS[0];
		this->bin_param_dim = _pPARAMS[3];
		this->bin_param_outnum = _pPARAMS[2];
		delete r;
	}

	layer_param() {
	}

	~layer_param() {

		map<char_t, param>::iterator it;
		for (it = data.begin(); it != data.end(); ++it) {
			for (int i = 0; i < it->second.size(); i++)
				vec_t().swap(it->second[i]);
			param().swap(it->second);
		}
		map<char_t, bin_param>::iterator b_it;
		for (b_it = bin_data.begin(); b_it != bin_data.end(); ++b_it) {
			for (int i = 0; i < b_it->second.size(); i++)
				vec_i().swap(b_it->second[i]);
			bin_param().swap(b_it->second);
		}
		map<char_t, bin_param>().swap(bin_data);
		map<char_t, param>().swap(data);
		map<char_t, int>().swap(param_outnum);
		map<char_t, int>().swap(param_dim);
		map<char_t, int>().swap(bin_param_outnum);
		map<char_t, int>().swap(bin_param_dim);

	}

	int caculate_space() {
		int sum = 0;
		if (this) {
			map<char_t, param>::iterator it;
			for (it = data.begin(); it != data.end(); ++it) {
				for (int i = 0; i < it->second.size(); i++) {
					sum += it->second[i].size();
				}
			}

			map<char_t, bin_param>::iterator it_;
			for (it_ = bin_data.begin(); it_ != bin_data.end(); ++it_) {
				for (int i = 0; i < it_->second.size(); i++) {
					sum += it_->second[i].size();
				}
			}
		}
		return sum;
	}

	void _RESET_DATA() {

		if (this) {
			map<char_t, param>::iterator it;
			for (it = data.begin(); it != data.end(); ++it) {
				for (int n = 0; n < it->second.size(); n++) {
					for (int i = 0; i < it->second[0].size(); i++)
						it->second[n][i] = 0.0;
				}
			}

			map<char_t, bin_param>::iterator it_;
			for (it_ = bin_data.begin(); it_ != bin_data.end(); ++it_) {
				for (int n = 0; n < it_->second.size(); n++) {
					for (int i = 0; i < it_->second[0].size(); i++)
						it_->second[n][i] = 0;
				}
			}
		}
	}

	//for normal data
	map<char_t, param> data;
	//for binary data
	map<char_t, bin_param> bin_data;

#endif

	map<char_t, int> param_outnum;
	map<char_t, int> param_dim;

	map<char_t, int> bin_param_outnum;
	map<char_t, int> bin_param_dim;

private:

};

}
;
