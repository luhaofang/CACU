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

#include <string.h>

namespace mycnn {

class sgd {

public:

	sgd(network *&net) {

		assert(net->net_.size() != 0);
		this->net = net;
		char_t param_name;
		char_t layer_name;
		map<char_t, int>::iterator it;
		for (int i = 0; i < this->net->layers.size(); i++) {
			layer_name = this->net->layers[i];
			pP_SPACE _s_params = this->net->net_[layer_name]->_pPARAMS;

			layer_param* _param = new layer_param(_s_params);
			layer_param* _acc_param = new layer_param(_s_params);
			this->data_v[layer_name] = _param;
			this->data_acc_v[layer_name] = _acc_param;

		}

		lr = (float_t) LEARN_RATE;
	}

	~sgd() {
		DELETE_SPACE();
		map<char_t, layer_param*>().swap(data_v);
		map<char_t, layer_param*>().swap(data_acc_v);
	}

	void caculate_sgd_data_space() {

		int sum = 0;
		map<char_t, layer_param*>::iterator it;
		for (it = data_v.begin(); it != data_v.end(); ++it) {
			sum += it->second->caculate_space();
			sum += data_acc_v[it->first]->caculate_space();
			printf("%s layer sgd space require : %d \n", it->first.c_str(),
					it->second->caculate_space());
		}
		printf("space costs : %d mb\n", sum * sizeof(float) / 1024 / 1024);
	}

	void reset_sgd_space() {

		map<char_t, layer_param*>::iterator it;
		for (it = data_v.begin(); it != data_v.end(); ++it) {
			it->second->_RESET_DATA();
		}
	}

	void train(int iter) {
		clock_t start;
		clock_t _start;
		clock_t end;
		_start = clock();

		net->set_iter(iter);
		net->set_phrase(type::train);

		net->reset_data_space();

		reset_sgd_space();

		for (int i = 0; i < this->net->layers.size(); i++) {
			//printf("==================%s\n",this->net->layers[i]);
			//start = clock();
			this->net->net_[this->net->layers[i]]->forward();
			//end = clock();
			//printf("%s layer forward time cost: %d ms\n",
			//		this->net->layers[i].c_str(), (end - start) / 1000);
		}

		for (int i = this->net->layers.size() - 1; i >= 0; i--) {
			//start = clock();
			this->net->net_[this->net->layers[i]]->backward(
					data_v[this->net->layers[i]]);
			//end = clock();
			//printf("%s layer backward time cost: %d ms\n",
			//		this->net->layers[i].c_str(), (end - start) / 1000);
		}

		update_params();

		end = clock();

		printf("%d/iter ms  lr : %f\n", (end - _start) / 1000, this->lr);

		if (iter % STEP_SIZE == STEP_SIZE-1) {
			this->lr = this->lr * 0.1;
		}
	}

	network *net;

private:

	void DELETE_SPACE() {

		map<char_t, layer_param*>::iterator it;
		for (it = data_v.begin(); it != data_v.end(); ++it) {
			delete it->second;
			it->second = NULL;
		}

		for (it = data_acc_v.begin(); it != data_acc_v.end(); ++it) {
			delete it->second;
			it->second = NULL;
		}
	}

#if GPU_MODE

	void update_params()
	{
		//need kernel function
		float_t momentum = (float_t)MOMENTUM;
		float_t weight_decay = (float_t)WEIGHT_DECAY;
		float_t local_rate;

		cudaError_t res;

		map<char_t, layer_param*>::iterator it;
		for (it = data_v.begin(); it != data_v.end(); ++it) {
			char_t layer_name = it->first;
			//cauclate acc_v;
			map<char_t, float_t **>::iterator _it;
			for (_it = it->second->data.begin(); _it != it->second->data.end();
					++_it) {
				char_t param_name = _it->first;
				if (param_name == "w" )
				local_rate = this->lr * net->net_[layer_name]->lr_w;
				else
				local_rate = this->lr * net->net_[layer_name]->lr_b;
//
//				vec_t test_data(1);
//
//				res = cudaMemcpy((void*) (&test_data[0]), (void*) (data_v[layer_name]->s_data[param_name]),
//						test_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
//				CHECK(res);
//				printf("%s_%s: ",layer_name.c_str(),param_name.c_str());
//
//				printf("dw_%.10f,",test_data[0]);

				CACU_SCALE_GPU_A(data_acc_v[layer_name]->data[param_name],momentum,data_acc_v[layer_name]->param_outnum[param_name],
						data_acc_v[layer_name]->param_dim[param_name],data_acc_v[layer_name]->data[param_name],0);

				CACU_AXBY_GPU(net->net_[layer_name]->params->data[param_name], weight_decay , data_acc_v[layer_name]->param_outnum[param_name], data_acc_v[layer_name]->param_dim[param_name],
						_it->second, (float_t)1.0,_it->second);

				CACU_AXBY_GPU(_it->second,local_rate,data_acc_v[layer_name]->param_outnum[param_name],data_acc_v[layer_name]->param_dim[param_name],
						data_acc_v[layer_name]->data[param_name],(float_t)1,data_acc_v[layer_name]->data[param_name]);

				if (param_name == "real_w")
				CACU_AXBY_CROP_GPU(net->net_[layer_name]->params->data[param_name],(float_t)1.0,data_acc_v[layer_name]->param_outnum[param_name],
						data_acc_v[layer_name]->param_dim[param_name],data_acc_v[layer_name]->data[param_name],(float_t)(-1.0),net->net_[layer_name]->params->data[param_name]);
				else
				CACU_AXBY_GPU(net->net_[layer_name]->params->data[param_name],(float_t)1.0,data_acc_v[layer_name]->param_outnum[param_name],
						data_acc_v[layer_name]->param_dim[param_name],data_acc_v[layer_name]->data[param_name],(float_t)(-1.0),net->net_[layer_name]->params->data[param_name]);

//				res = cudaMemcpy((void*) (&test_data[0]), (void*) (data_acc_v[layer_name]->s_data[param_name]),
//						test_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
//				CHECK(res);
//
//				printf("v_%.10f,",test_data[0]);
//
//				res = cudaMemcpy((void*) (&test_data[0]), (void*) (net->net_[layer_name]->params->s_data[param_name]),
//						test_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
//				CHECK(res);
//
//				printf("w_%.10f,",test_data[0]);
//
//				printf("\n");

			}
		}
	}

#else

	void update_params() {
		float_t *pv, *p, *pw, local_rate;
		float_t momentum = (float_t) MOMENTUM;
		float_t weight_decay = (float_t) WEIGHT_DECAY;

		map<char_t, layer_param*>::iterator it;
		for (it = data_v.begin(); it != data_v.end(); ++it) {
			char_t layer_name = it->first;
			//cauclate acc_v;
			map<char_t, vector<vec_t>>::iterator _it;
			for (_it = it->second->data.begin(); _it != it->second->data.end();
					++_it) {
				char_t param_name = _it->first;
				if (param_name == "w")
					local_rate = lr * net->net_[layer_name]->lr_w;
				else
					local_rate = lr * net->net_[layer_name]->lr_b;
				//num's iteration
				for (int num = 0; num < _it->second.size(); num++) {
					//gradient
					pw = &_it->second[num][0];
					//param
					p =
							&net->net_[layer_name]->params->data[param_name][num][0];
					//v
					pv = &data_acc_v[layer_name]->data[param_name][num][0];
					//if (num == 0)
					//	printf("layer('%s')_%s : dw_%.10f,", layer_name.c_str(),
					//			param_name.c_str(), *pw);
					//update params and acc_v
					for (int length = 0; length < _it->second[num].size();
							length++) {

						*(pv + length) = momentum * (*(pv + length))
								+ local_rate
										* (weight_decay * (*(p + length))
												+ (*(pw + length)));
						if (param_name == "real_w") {
							if (abs(*(p + length) + *(pv + length)) <= 1)
								*(p + length) -= *(pv + length);
						} else {
							*(p + length) -= *(pv + length);
						}
					}
					//if (num == 0)
					//	printf("v_%.10f, w_%.10f\n", *pv, *p);
				}
			}
		}
	}

#endif

	//for dv,erarse every iteration
	map<char_t, layer_param*> data_v;
	//for accumulated dv(update params)
	map<char_t, layer_param*> data_acc_v;

	float_t lr;

};

}
;
