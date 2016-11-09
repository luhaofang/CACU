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

class bin_activation_layer: public layer {
	activation::sign h_;

public:

	bin_activation_layer(char_t layer_name, int input_dim, int channel,
			int kernel_size, int stride, int pad, type phrase, float_t lr_w =
					1.0, float_t lr_b = 1.0) :
			layer(layer_name, input_dim, channel, kernel_size, stride, pad,
					phrase, lr_w, lr_b) {
		this->layer_name = layer_name;
		this->output_dim = (input_dim + pad * 2 - kernel_size) / stride + 1;
		this->stride = stride;
		this->pad = pad;
		this->kernel_size = kernel_size;
		this->output_channel = channel;
		this->input_dim = input_dim;
		this->channel = channel;
		this->phrase = phrase;
		this->set_lr_w(lr_w);
		this->set_lr_b(lr_b);
		INIT_SPACE_DATA();
		INIT_PARAM_SAPCE_DATA();
		INIT_STORAGE_DATA();
	}

#if GPU_MODE

	virtual const void forward() override
	{

		//sum by channel
		CACU_SUM_ABS_GPU(bottoms[0]->data, bottoms[0]->num, input_dim*input_dim*channel,
				input_dim*input_dim, channel, storage_data->data["sum_k"]);

		//mean by channel and scaled by 1.0/kernel_size*kernel_size
		CACU_SCALE_GPU_A(storage_data->data["sum_k"],(float_t)1.0 / (channel*kernel_size*kernel_size),bottoms[0]->num ,input_dim*input_dim, storage_data->data["sum_k"], 0);

		copy_padding_data_blob_gpu(storage_data->s_data["sum_k"], bottoms[0]->num, input_dim, 1, pad, storage_data->s_data["sum_pad_k"]);

		//pad for convoluation
		img2col_gpu(storage_data->s_data["sum_pad_k"], bottoms[0]->num, 1, input_dim+2*pad, kernel_size, stride, output_dim, storage_data->s_data["ks"]);

		CACU_SUM_SIZE_GPU(storage_data->data["ks"],bottoms[0]->num, (1 * kernel_size*kernel_size), output_dim*output_dim*kernel_size*kernel_size,
				output_dim*output_dim,tops[0]->data);

		BIT_CACU_SIGN_GPU(bottoms[0]->data, bin_tops[0]->bin_data,bottoms[0]->num,input_dim*input_dim*channel);

	}

	virtual const void backward(layer_param *&v) override
	{
		//caculate design(I)
		BIT_CACU_DESIGN_GPU(bin_tops[0]->diff, storage_data->data["df"],bin_tops[0]->num,input_dim*input_dim*channel);
		//scaled by design(I)
		CACU_SCALE_GPU_D(bin_tops[0]->diff, storage_data->data["df"],bin_tops[0]->num,input_dim*input_dim*channel, bottoms[0]->diff);
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
				sum += (tops[i]->num*input_dim*input_dim*channel);
				if (train == phrase)
				sum += (tops[i]->num*input_dim*input_dim*channel);
			}
		}
		for (int i = 0; i < bin_tops.size(); i++) {
			if (bin_tops[i]->num > 0) {
				sum += (bin_tops[i]->num*input_dim*input_dim*channel / BIN_SIZE);
				if (train == phrase)
				sum += (bin_tops[i]->num*input_dim*input_dim*channel);
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
		//sum by channel
		CACU_SUM_ABS_CPU(bottoms[0]->data, channel, storage_data->data["sum_k"]);
		//mean by channel and scaled by 1.0/kernel_size*kernel_size
		for (int num = 0; num < bottoms[0]->data.size(); num++) {
			CACU_SCALE_CPU(storage_data->data["sum_k"][num], (float_t)1.0 / (channel*kernel_size*kernel_size), storage_data->data["sum_k"][num], 0);
		}
		//pad for convoluation
		img2col(storage_data->data["sum_k"], kernel_size, stride, pad, input_dim, output_dim, storage_data->data["ks"]);

		for (int num = 0; num < bottoms[0]->data.size(); num++) {
			CACU_SUM_SIZE_CPU(storage_data->data["ks"][num], (1 * kernel_size*kernel_size), tops[0]->data[num]);
		}

		BIT_CACU_SIGN(bottoms[0]->data, bin_tops[0]->bin_data);

	}

	virtual const void backward(layer_param *&v) override
	{
		//caculate design(I)
		BIT_CACU_DESIGN(bin_tops[0]->diff, storage_data->data["df"]);
		//scaled by design(I)
		CACU_SCALE_CPU(bin_tops[0]->diff, storage_data->data["df"], bottoms[0]->diff);

		//copy_data(bin_tops[0]->diff, bottoms[0]->diff, 0);

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

		//for signed(I)
		_bin_param_outnum.push_back(channel);
		_bin_param_dim.push_back(this->input_dim);

		//for K
		_param_outnum.push_back(1);
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
		//param_dim equals to _param_dim
		map<char_t, int> _param_outnum;
		map<char_t, int> _param_dim;

		map<char_t, int> _bin_param_outnum;
		map<char_t, int> _bin_param_dim;

		//here to initial the layer's params size
		////////////////////////////////////////
		_param_outnum["sum_k"] = BATCH_SIZE;
		_param_dim["sum_k"] = 1 * input_dim * input_dim;

		_param_outnum["ks"] = BATCH_SIZE;
		_param_dim["ks"] = 1 * output_dim * kernel_size * output_dim
				* kernel_size;

		if (GPU_MODE) {
			_param_outnum["sum_pad_k"] = BATCH_SIZE;
			_param_dim["sum_pad_k"] = 1 * (input_dim+2*pad) * (input_dim+2*pad);
		}

		if (train == phrase) {

			_param_outnum["df"] = BATCH_SIZE;
			_param_dim["df"] = output_channel * input_dim * input_dim;

		}

		////////////////////////////////////////

		_pSTORAGE.push_back(_param_outnum);
		_pSTORAGE.push_back(_param_dim);
		_pSTORAGE.push_back(_bin_param_outnum);
		_pSTORAGE.push_back(_bin_param_dim);
	}

	~bin_activation_layer() {

	}

private:

};

}
;
