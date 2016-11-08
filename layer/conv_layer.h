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

class conv_layer: public layer {

public:

	conv_layer(char_t layer_name, int input_dim, int channel,
			int output_channel, int kernel_size, int stride, int pad,
			type phrase = train, float_t lr_w = 1.0, float_t lr_b = 1.0) :
			layer(layer_name, input_dim, channel, output_channel, kernel_size,
					stride, pad, phrase, lr_w, lr_b) {
		this->layer_name = layer_name;
		this->output_dim = (input_dim + pad * 2 - kernel_size) / stride + 1;
		this->stride = stride;
		this->pad = pad;
		this->kernel_size = kernel_size;
		this->channel = channel;
		this->input_dim = input_dim;
		this->output_channel = output_channel;
		this->set_lr_w(lr_w);
		this->set_lr_b(lr_b);
		this->phrase = phrase;
		INIT_SPACE_DATA();
		INIT_PARAM_SAPCE_DATA();
		INIT_STORAGE_DATA();
	}

#if GPU_MODE

	virtual const void forward() override
	{
		clock_t start = clock();
		copy_padding_data_blob_gpu(bottoms[0]->s_data, bottoms[0]->num, input_dim, channel, pad, storage_data->s_data["pad_data"]);
		clock_t end = clock();
		//printf("%d,",end-start);

		start = clock();
		//unroll for convoluation
		img2col_gpu( storage_data->s_data["pad_data"], bottoms[0]->num, channel,
				input_dim+2*pad, kernel_size, stride, output_dim,storage_data->s_data["col_data"]);
		end = clock();
		//printf("%d,",end-start);

		start = clock();
		CACU_SCALE_SUM_ROW_GPU(storage_data->s_data["col_data"], bottoms[0]->num, kernel_size*kernel_size*channel, output_channel, output_dim*output_dim*output_channel, params->s_data["w"],
				params->s_data["bias"], tops[0]->s_data);
		end = clock();
		//printf("%d\n",end-start);

	}

	virtual const void backward(layer_param *&v) override
	{

		CACU_DECONV_W_B_GPU(storage_data->s_data["col_data"], tops[0]->s_diff,tops[0]->num, kernel_size, output_channel, output_dim, channel,stride, v->s_data["w"],v->s_data["bias"]);

		reset_data_gpu( storage_data->s_data["pad_data"],bottoms[0]->num,(input_dim+2*pad)*(input_dim + 2*pad)*channel);

		reset_data_gpu( storage_data->s_data["col_data"],bottoms[0]->num,(output_dim*kernel_size)*(output_dim * kernel_size)*channel);

		//CACU_DECONV_DIFF_GPU(params->data["w"], tops[0]->diff,kernel_size, output_channel, tops[0]->num,input_dim, pad, channel, stride, storage_data->data["pad_data"]);

		CACU_DECONV_DIFF_COL_GPU(params->s_data["w"], tops[0]->s_diff,kernel_size, output_channel, tops[0]->num,input_dim, pad, channel, stride, storage_data->s_data["col_data"]);

		col2img_gpu(storage_data->s_data["col_data"], bottoms[0]->num, channel, input_dim+2*pad, kernel_size, stride, output_dim, storage_data->s_data["pad_data"]);

		copy_unpadding_data_gpu(storage_data->s_data["pad_data"], bottoms[0]->num, input_dim, channel, pad, bottoms[0]->s_diff);
	}

	virtual const void save(std::ostream& os) override {
		cudaError_t res;

		for (int i = 0; i < layer_name.size(); i++)
		{
			os.write((char*)(&layer_name[i]), sizeof(layer_name[i]));
		}

		vec_t _data(output_channel*channel*kernel_size*kernel_size);

		res = cudaMemcpy((void*) (&_data[0]), (void*) (params->s_data["w"]),
				_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
		CHECK(res);

		for (auto w : _data) os.write((char*)(&w), sizeof(w));

		_data.resize(output_channel);

		res = cudaMemcpy((void*) (&_data[0]), (void*) (params->s_data["bias"]),
				_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
		CHECK(res);

		for (auto w : _data) os.write((char*)(&w), sizeof(w));

	}

	virtual const void load(std::ifstream& is) override {

		string _sdata;
		char _c;
		float_t _d;

		for (int i = 0; i < layer_name.size(); i++)
		{
			is.read(&_c, 1);
			_sdata += _c;
		}

		cudaError_t res;

		assert(_sdata == layer_name);

		vec_t _data(output_channel*channel*kernel_size*kernel_size);

		for (int num = 0; num < params->param_outnum["w"]; num++)
		for (int k = 0; k < params->param_dim["w"]; k++)
		{
			is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
			_data[num*params->param_dim["w"]+k] = _d;
		}
		res = cudaMemcpy((void*) (params->s_data["w"]), (void*)(&_data[0]) ,
				_data.size() * sizeof(float_t), cudaMemcpyHostToDevice);
		CHECK(res);

		_data.resize(output_channel);

		for (int num = 0; num < params->param_outnum["bias"]; num++)
		for (int k = 0; k < params->param_dim["bias"]; k++)
		{
			is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
			_data[num*params->param_dim["bias"]+k] = _d;
		}
		res = cudaMemcpy((void*) (params->s_data["bias"]), (void*)(&_data[0]) ,
				_data.size() * sizeof(float_t), cudaMemcpyHostToDevice);
		CHECK(res);

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

		//unroll for convoluation
		img2col(bottoms[0]->data, kernel_size, stride, pad, input_dim, output_dim, storage_data->data["col_data"]);

		for (int num = 0; num < bottoms[0]->data.size(); num++) {

			CACU_SCALE_SUM_ROW_CPU(storage_data->data["col_data"][num], kernel_size*kernel_size*channel, params->data["w"], params->data["bias"], tops[0]->data[num]);

		}

	}

	//by CPU
	virtual const void backward(layer_param *&v) override
	{

		for (int num = 0; num < bottoms[0]->data.size(); num++) {

			CACU_DECONV_W_CPU(storage_data->data["col_data"][num], tops[0]->diff[num], kernel_size, input_dim, pad, channel, stride, v->data["w"], v->data["bias"]);
		}

		CACU_RESET_CPU(storage_data->data["col_data"]);

		for(int num = 0; num < bottoms[0]->num;num++)
		{
			CACU_DECONV_DIFF_CPU(params->data["w"], tops[0]->diff[num], kernel_size, channel, output_dim, storage_data->data["col_data"][num]);
		}

		CACU_RESET_CPU(storage_data->data["pad_data"]);

		col2img(storage_data->data["col_data"], kernel_size, stride, pad, input_dim, output_dim, storage_data->data["pad_data"]);

		copy_unpadding_data(storage_data->data["pad_data"], input_dim, pad, bottoms[0]->diff);

	}

	virtual const void save(std::ostream& os) override {

		for (int i = 0; i < layer_name.size(); i++)
		{
			os.write((char*)(&layer_name[i]), sizeof(layer_name[i]));
		}

		for (auto ws : params->data["w"]) {
			for (auto w : ws) os.write((char*)(&w), sizeof(w));
		}
		for (auto ws : params->data["bias"]) {
			for (auto w : ws) os.write((char*)(&w), sizeof(w));
		}

	}

	virtual const void load(std::ifstream& is) override {

		string _data;
		char _c;
		float_t _d;

		for (int i = 0; i < layer_name.size(); i++)
		{
			is.read(&_c, 1);
			_data += _c;
		}

		assert(_data == layer_name);

		for (int num = 0; num < params->data["w"].size(); num++)
		for (int k = 0; k < params->data["w"][0].size(); k++)
		{
			is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
			params->data["w"][num][k] = _d;
		}

		for (int num = 0; num < params->data["bias"].size(); num++)
		for (int k = 0; k < params->data["bias"][0].size(); k++)
		{
			is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
			params->data["bias"][num][k] = _d;
		}
	}

	virtual const void setup() override {

	}

	virtual const int caculate_data_space() override {

		//printf("assert %s layer input %d == %d failed!", layer_name, bottoms[0]->data[0].size(), channel*input_dim*input_dim);

		//real I
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

		_param_outnum["w"] = output_channel;
		_param_dim["w"] = kernel_size*kernel_size*channel;

		_param_outnum["bias"] = output_channel;
		_param_dim["bias"] = 1;

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

		_param_outnum.push_back(output_channel);
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

		_param_outnum["col_data"] = BATCH_SIZE;
		_param_dim["col_data"] = output_dim * output_dim * channel * kernel_size
				* kernel_size;

		if (train == phrase) {
			_param_outnum["pad_data"] = BATCH_SIZE;
			_param_dim["pad_data"] = channel * (input_dim + 2 * pad)
					* (input_dim + 2 * pad);
		}
		////////////////////////////////////////

		_pSTORAGE.push_back(_param_outnum);
		_pSTORAGE.push_back(_param_dim);
		_pSTORAGE.push_back(_bin_param_outnum);
		_pSTORAGE.push_back(_bin_param_dim);
	}

	~conv_layer() {

	}

private:

	void DELETE_DATA() {

	}

};

}
;
