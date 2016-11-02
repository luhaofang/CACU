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

#if GPU_MODE
#include <cuda_runtime.h>

#endif

namespace mycnn {

class softmax_layer: public layer {

public:

	softmax_layer(char_t layer_name, int input_dim, type phrase, float_t lr_w =
			1.0, float_t lr_b = 1.0) :
			layer(layer_name, input_dim, phrase, lr_w, lr_b) {
		this->layer_name = layer_name;
		this->output_dim = input_dim;
		this->input_dim = input_dim;
		this->phrase = phrase;
		this->channel = 1;
		this->kernel_size = 0;
		this->output_channel = 1;
		this->pad = 0;
		this->stride = 0;
		this->set_lr_w(lr_w);
		this->set_lr_b(lr_b);
		INIT_SPACE_DATA();
		INIT_PARAM_SAPCE_DATA();
		INIT_STORAGE_DATA();

		if (train == this->phrase) {
			os.open("/home/seal/dataset/experiment/cifar10/loss.txt");
			os.precision(std::numeric_limits<float_t>::digits10);
		}
	}

#if GPU_MODE

	virtual const void forward() override
	{

		CACU_SOFTMAX_GPU(bottoms[0]->data, bottoms[0]->num, input_dim ,tops[0]->data);

		cudaError_t res;

		float_t index = 0;

		int label;

		if (train == this->phrase)
		{

			loss = 0.0;

			CACU_CE_LOSS_GPU(tops[0]->data, bottoms[1]->data, bottoms[0]->num, cuda_loss);

			res = cudaMemcpy((void*) (&loss), (void*) (cuda_loss), sizeof(float_t), cudaMemcpyDeviceToHost);
			CHECK(res);

			loss = loss / (float_t)bottoms[0]->num;

			os << loss << "\n" << flush;

			printf("iter_%d 	loss:%f\n",iter, loss);

		}
	}

	virtual const void backward(layer_param *&v) override
	{

		copy_data_gpu(tops[0]->data, bottoms[0]->diff, bottoms[0]->num, input_dim,0);

		CACU_SUB_INDEX_GPU(bottoms[0]->diff, bottoms[1]->data,float_t(1), bottoms[0]->num, bottoms[0]->diff);

		CACU_SCALE_GPU_A(bottoms[0]->diff, (float_t)(1.0 / bottoms[0]->num),bottoms[0]->num, input_dim, bottoms[0]->diff, 0);

		CACU_SCALE_GPU_A(bottoms[0]->diff, loss_weight,bottoms[0]->num, input_dim, bottoms[0]->diff, 0);

	}

	virtual const void save(std::ostream& os) override {

	}

	virtual const void load(std::ifstream& is) override {

	}

	virtual const void setup() override {

		cudaError_t res;
		res = cudaMalloc((void**) (&cuda_loss), sizeof(float_t));
		CHECK(res);

	}

	virtual const int caculate_data_space() override {

		printf("%s top costs : %d \n", layer_name.c_str(), 0);

		printf("%s params costs %d \n", layer_name.c_str(), params->caculate_space());

		return 0;
	}

#else

	virtual const void forward() override
	{
		float_t *sp, max_, *p, sum;
		for (int num = 0; num < bottoms[0]->data.size(); num++)
		{
			sp = &tops[0]->data[num][0];
			max_ = bottoms[0]->data[num][0];
			p = &bottoms[0]->data[num][0];
			sum = 0.0;
			for (int i = 0; i < input_dim; i++)
			max_ = max(*(p + i), max_);
			for (int i = 0; i < input_dim; i++)
			{
				*(sp + i) = exp(*(p + i) - max_);
				sum += *(sp + i);
			}
			for (int i = 0; i < input_dim; i++)
			{
				*(sp + i) = (*(sp + i) / sum);
			}
		}

		float_t index = 0;

		int label;

		if (train == this->phrase)
		{
			this->loss = 0.0;
			for (int num = 0; num < bottoms[0]->data.size(); num++)
			{
				label = (unsigned int)bottoms[1]->data[num][0];
				loss -= log(tops[0]->data[num][label]);
			}
			loss = loss / (float_t)bottoms[0]->data.size();

			printf("iter_%d 	loss:%f\n",iter, loss);
//			printf("==============================\n");
//			printf("loss:%.10f\n", loss);
//			printf("==============================\n");
//
//			os << loss << "\n";
//			if (BATCH_SIZE >=3 )
//			printf("sample1_%d:%.10f,sample2_%d:%.10f,sample3_%d:%.10f\n", (int)bottoms[1]->data[0][0], tops[0]->data[0][(int)bottoms[1]->data[0][0]], (int)bottoms[1]->data[1][0], tops[0]->data[1][(int)bottoms[1]->data[1][0]], (int)bottoms[1]->data[2][0], tops[0]->data[2][(int)bottoms[1]->data[2][0]]);
//			else
//			printf("sample1_%d:%.10f\n", (int)bottoms[1]->data[0][0], tops[0]->data[0][(int)bottoms[1]->data[0][0]]);
		}
	}

	virtual const void backward(layer_param *&v) override
	{

		copy_data(tops[0]->data, bottoms[0]->diff, 0);

		for (int num = 0; num < tops[0]->data.size(); num++) {

			int label = (int)bottoms[1]->data[num][0];

			bottoms[0]->diff[num][label] -= float_t(1);

			CACU_SCALE_CPU(bottoms[0]->diff[num], (float_t)1 / bottoms[0]->diff.size(), bottoms[0]->diff[num], 0);

			CACU_SCALE_CPU(bottoms[0]->diff[num], loss_weight, bottoms[0]->diff[num], 0);

		}
	}

	virtual const void save(std::ostream& os) override {

	}

	virtual const void load(std::ifstream& is) override {

	}

	virtual const void setup() override {

	}

	virtual const int caculate_data_space() override {

		assert(bottoms[0]->data[0].size() == output_dim);

		int sum = 0;
		for (int i = 0; i < tops.size(); i++) {
			if (tops[i]->data.size() > 0) {
				sum += (tops[i]->data.size()*tops[i]->data[0].size());
				if (train == phrase)
				sum += tops[i]->diff.size()*tops[i]->diff[0].size();
			}
		}
		for (int i = 0; i < bin_tops.size(); i++) {
			if (bin_tops[i]->bin_data.size() > 0) {
				sum += (bin_tops[i]->bin_data.size()*bin_tops[i]->bin_data[0].size() / BIN_SIZE);
				if (train == phrase)
				sum += bin_tops[i]->diff.size()*bin_tops[i]->diff[0].size();
			}
		}

		printf("%s top costs : %d \n", layer_name.c_str(), sum);

		sum += params->caculate_space();
		sum += storage_data->caculate_space();
		printf("%s params costs %d \n", layer_name.c_str(), params->caculate_space());

		return sum;
	}

#endif

	virtual const void INIT_PARAM_SAPCE_DATA() override {
		//param_dim equals to _param_dim
		map<char_t, int> _param_outnum;
		map<char_t, int> _param_dim;

		map<char_t, int> _bin_param_outnum;
		map<char_t, int> _bin_param_dim;

		//here to initial the layer's params size
		////////////////////////////////////////

		////////////////////////////////////////

		_pPARAMS.push_back(_param_outnum);
		_pPARAMS.push_back(_param_dim);
		_pPARAMS.push_back(_bin_param_outnum);
		_pPARAMS.push_back(_bin_param_dim);
	}

	virtual const void INIT_SPACE_DATA() override {

		//param_dim equals to channel * dim * dim
		vector<int> _param_outnum;
		vector<int> _param_dim;

		vector<int> _bin_param_outnum;
		vector<int> _bin_param_dim;

		//here to initial the layer's space size
		////////////////////////////////////////

		_param_outnum.push_back(output_dim);
		_param_dim.push_back(1);

		////////////////////////////////////////

		for (int i = 0; i < _param_dim.size(); i++) {
			blob *top;
			tops.push_back(top);
		}

		for (int i = 0; i < _bin_param_dim.size(); i++) {
			bin_blob *bin_top;
			bin_tops.push_back(bin_top);
		}

		_PARAMS.push_back(_param_outnum);
		_PARAMS.push_back(_param_dim);
		_PARAMS.push_back(_bin_param_outnum);
		_PARAMS.push_back(_bin_param_dim);
	}

	virtual const void INIT_STORAGE_DATA() {
		//param_dim equals to _param_dim
		map<char_t, int> _param_outnum;
		map<char_t, int> _param_dim;

		map<char_t, int> _bin_param_outnum;
		map<char_t, int> _bin_param_dim;

		//here to initial the layer's params size
		////////////////////////////////////////

		////////////////////////////////////////

		_pSTORAGE.push_back(_param_outnum);
		_pSTORAGE.push_back(_param_dim);
		_pSTORAGE.push_back(_bin_param_outnum);
		_pSTORAGE.push_back(_bin_param_dim);
	}

	~softmax_layer() {
		os.close();

	}

private:

	float_t loss;

	float_t loss_weight = 1;

	std::ofstream os;

#if GPU_MODE
	float_t *cuda_loss;
#endif

};

}
;
