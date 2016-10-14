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

namespace mycnn {

class average_pooling_layer: public layer {

public:

	average_pooling_layer(char_t layer_name, int input_dim, int channel,
			int kernel_size, int stride, type phrase, float_t lr_w = 1.0,
			float_t lr_b = 1.0) :
			layer(layer_name, input_dim, channel, kernel_size, stride, phrase,
					lr_w, lr_b) {
		this->layer_name = layer_name;
		this->kernel_size = kernel_size;
		this->channel = channel;
		this->input_dim = input_dim;
		this->output_channel = channel;
		this->phrase = phrase;
		this->stride = stride;
		if ((input_dim - kernel_size) % stride != 0)
			this->pad = kernel_size - stride;
		this->output_dim = (input_dim + pad - kernel_size) / stride + 1;
		this->set_lr_w(lr_w);
		this->set_lr_b(lr_b);

		INIT_SPACE_DATA();
		INIT_PARAM_SAPCE_DATA();
		INIT_STORAGE_DATA();
	}

#if GPU_MODE

	virtual const void forward() override
	{

		CACU_A_POOLING_GPU(bottoms[0]->data, bottoms[0]->num,kernel_size, input_dim,output_dim, pad, output_dim*output_dim*output_channel, channel, stride,tops[0]->data);
	}

	virtual const void backward(layer_param *&v) override
	{
		copy2mean_gpu(tops[0]->diff, tops[0]->num, output_dim, input_dim,
				channel, kernel_size, stride, pad,bottoms[0]->diff);

//		cudaError_t res;
//		vec_t test_data(bottoms[0]->num*input_dim*input_dim*channel);
//		res = cudaMemcpy((void*) (&test_data[0]), (void*) (bottoms[0]->s_diff),
//				test_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
//		CHECK(res);
//		for(int i = 0; i < test_data.size(); i ++)
//		printf("%f,",test_data[i]);
//		printf("\n");
	}

	virtual const void save(std::ostream& os) override {

	}

	virtual const void load(std::ifstream& is) override {

	}

	virtual const void setup() override {

	}

	virtual const int caculate_data_space() override {

		assert(bottoms[0]->channel == channel);
		assert(bottoms[0]->dim == input_dim);

		int sum = 0;
		for (int i = 0; i < tops.size(); i++) {
			if (tops[i]->num > 0) {
				sum += (tops[i]->num*output_dim*output_dim*channel);
				if (train == phrase)
				sum += (tops[i]->num*output_dim*output_dim*channel);
			}
		}
		for (int i = 0; i < bin_tops.size(); i++) {
			if (bin_tops[i]->num > 0) {
				sum += (bin_tops[i]->num*output_dim*output_dim*channel / BIN_SIZE);
				if (train == phrase)
				sum += (bin_tops[i]->num*output_dim*output_dim*channel);
			}
		}

		printf("%s top costs : %d \n", layer_name.c_str(), sum);

		sum += params->caculate_space();
		sum += storage_data->caculate_space();
		printf("%s params costs %d \n", layer_name.c_str(), params->caculate_space());
	}

#else

	virtual const void forward() override
	{
		if (pad != 0) {
			//if input_dim is less than the output_dim,should be padding to caculate
			blob *new_input = new blob(bottoms[0]->num, channel, input_dim + pad);
				
			append_padding_data_blob(bottoms[0]->data, input_dim, pad, new_input->data);

			blob col_data(bottoms[0]->data.size(), channel, output_dim*kernel_size);

			img2col(new_input->data, kernel_size, stride, 0, input_dim + pad, output_dim, col_data.data);

			int param_w,param_h,flag = output_dim - 1;

			float_t k_size = (float_t)kernel_size*kernel_size;

			for (int num = 0; num < bottoms[0]->data.size(); num++) {

				CACU_SUM_SIZE_CPU(col_data.data[num], kernel_size*kernel_size, tops[0]->data[num]);

				CACU_SCALE_CPU(tops[0]->data[num], (1.0 / (float_t)(kernel_size*kernel_size)), tops[0]->data[num], 0);

				//fix
				for (int i = 0; i < output_dim; i++)
				{
					for (int j = 0; j < output_dim; j++)
					{
						param_w = kernel_size, param_h = kernel_size;
						if (i == flag)
						param_w = kernel_size - pad;

						if (j == flag)
						param_h = kernel_size - pad;

						if (param_w != kernel_size || param_h != kernel_size)
						for (int c = 0; c < channel; c++)
						{
							tops[0]->data[num][(i*output_dim + j)*channel + c] = tops[0]->data[num][(i*output_dim + j)*channel + c] * (k_size / (float_t)(param_h*param_w));
						}
					}
				}
			}
			delete new_input;
		}
		else
		{
			blob col_data(bottoms[0]->data.size(), channel, output_dim*kernel_size);

			img2col(bottoms[0]->data, kernel_size, stride, 0, input_dim, output_dim, col_data.data);

			for (int num = 0; num < bottoms[0]->data.size(); num++) {

				CACU_SUM_SIZE_CPU(col_data.data[num], kernel_size*kernel_size, tops[0]->data[num]);

				CACU_SCALE_CPU(tops[0]->data[num], (1.0 / (float_t)(kernel_size*kernel_size)), tops[0]->data[num], 0);

			}

		}
	}

	virtual const void backward(layer_param *&v) override
	{
		if (pad != 0) {

			blob new_input(bottoms[0]->diff.size(), channel, input_dim + pad);

			copy2mean(tops[0]->diff, output_dim, input_dim, channel, kernel_size, stride, pad, new_input.data);

			append_unpadding_data(new_input.data, input_dim, pad, bottoms[0]->diff);

		}
		else
		{
			//added to activied feature map
			copy2mean(tops[0]->diff, output_dim, input_dim, channel, kernel_size, stride, pad, bottoms[0]->diff);
		}

	}

	virtual const void save(std::ostream& os) override {

	}

	virtual const void load(std::ifstream& is) override {

	}

	virtual const void setup() override {

	}

	virtual const int caculate_data_space() override {

		assert(bottoms[0]->data[0].size() == channel*input_dim*input_dim);

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

		_param_outnum.push_back(this->output_channel);
		_param_dim.push_back(this->output_dim);

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

	~average_pooling_layer() {

	}

};

}
;
