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

class bin_conv_layer: public layer {
	activation::sign h_;

	activation::relu h;

public:

	bin_conv_layer(char_t layer_name, int input_dim, int channel,
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

		//for vec_i's block_size
		if (channel * kernel_size * kernel_size % BIN_SIZE == 0)
			block_size = (channel * kernel_size * kernel_size / BIN_SIZE);
		else
			block_size = (channel * kernel_size * kernel_size / BIN_SIZE + 1);

		INIT_SPACE_DATA();
		INIT_PARAM_SAPCE_DATA();
		INIT_STORAGE_DATA();
	}

#if GPU_MODE

	virtual const void forward() override
	{
		if (train == this->phrase)
		resigned_weights();

		copy_padding_data_sign_gpu(bin_bottoms[0]->s_bin_data, bottoms[0]->num,input_dim, channel, pad,pad_data->s_bin_data);
		//for img2col input data
		//pad for convoluation
		img2bitcol_gpu(pad_data->s_bin_data, bin_bottoms[0]->num, channel, input_dim, kernel_size, stride, pad, output_dim, storage_data->s_bin_data["pad_input"]);
		//printf("==================%s\n",this->layer_name);
		//caculate binaried convolution

		BIT_CACU_COUNT_CONV_GPU(storage_data->s_bin_data["pad_input"],params->s_bin_data["w"],bottoms[0]->s_data, params->s_data["a"], bottoms[0]->num,block_size,
				output_channel, output_dim*output_dim*output_channel,kernel_size*kernel_size*channel, tops[0]->s_data);

//		cudaError_t res;
//		vec_t test_data(block_size);
//		res = cudaMemcpy((void*) (&test_data[0]), (void*) (tops[0]->s_data),
//				test_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
//		CHECK(res);
//		for(int i = 0; i < test_data.size(); i ++)
//		printf("%d:%f,",i,test_data[i]);
//		printf("\n");
	}

	virtual const void backward(layer_param *&v) override
	{

		copy_padding_data_blob_gpu(bottoms[1]->s_data, bottoms[1]->num, input_dim, channel, pad, storage_data->s_data["pad_data"]);
		//for update weights
		img2col_gpu(storage_data->s_data["pad_data"], bottoms[1]->num, channel,
				input_dim+2*pad, kernel_size, stride, output_dim,storage_data->s_data["col_data"]);

		CACU_DECONV_W_BIN_GPU(storage_data->s_data["col_data"], tops[0]->s_diff,params->s_data["a"],tops[0]->num, kernel_size, output_channel, output_dim, channel,stride, v->s_data["real_w"]);

		reset_data_gpu( storage_data->s_data["col_data"],bottoms[0]->num,(output_dim*kernel_size)*(output_dim * kernel_size)*channel);

		CACU_DECONV_DIFF_COL_GPU(params->s_data["real_w"], tops[0]->s_diff,kernel_size, output_channel, tops[0]->num,input_dim, pad, channel, stride, storage_data->s_data["col_data"]);

		reset_data_gpu( storage_data->s_data["pad_data"],bottoms[0]->num,(input_dim+2*pad)*(input_dim + 2*pad)*channel);

		col2img_gpu(storage_data->s_data["col_data"], bottoms[0]->num, channel, input_dim+2*pad, kernel_size, stride, output_dim, storage_data->s_data["pad_data"]);

		copy_unpadding_data_gpu(storage_data->s_data["pad_data"], bottoms[0]->num, input_dim, channel, pad, bin_bottoms[0]->s_diff);

	}

	virtual const void save(std::ostream& os) override {
		cudaError_t res;

		for (int i = 0; i < layer_name.size(); i++)
		{
			os.write((char*)(&layer_name[i]), sizeof(layer_name[i]));
		}

		vec_i _bin_data(output_channel*block_size);
		res = cudaMemcpy((void*) (&_bin_data[0]), (void*) (params->s_bin_data["w"]),
				_bin_data.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		CHECK(res);

		for (auto w : _bin_data) os.write((char*)(&w), sizeof(w));

		vec_t _data(output_channel);

		res = cudaMemcpy((void*) (&_data[0]), (void*) (params->s_data["a"]),
				_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
		CHECK(res);

		for (auto w : _data) os.write((char*)(&w), sizeof(w));

		if(use_for_training) {

			_data.resize(output_channel*channel*kernel_size*kernel_size);

			res = cudaMemcpy((void*) (&_data[0]), (void*) (params->s_data["real_w"]),
					_data.size() * sizeof(float_t), cudaMemcpyDeviceToHost);
			CHECK(res);

			for (auto w : _data) os.write((char*)(&w), sizeof(w));
		}
	}

	virtual const void load(std::ifstream& is) override {

		string _sdata;
		char _c;
		float_t _d;
		unsigned int _u;

		for (int i = 0; i < layer_name.size(); i++)
		{
			is.read(&_c, 1);
			_sdata += _c;
		}

		cudaError_t res;

		assert(_sdata == layer_name);

		vec_i _bin_data(output_channel*block_size);

		for (int num = 0; num < params->bin_param_outnum["w"]; num++)
		for (int k = 0; k < block_size; k++)
		{
			is.read(reinterpret_cast<char*>(&_u), sizeof(unsigned int));
			_bin_data[num*block_size+k] = _u;
		}
		res = cudaMemcpy((void*) (params->s_bin_data["w"]), (void*)(&_bin_data[0]) ,
				_bin_data.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
		CHECK(res);

		vec_t _data(output_channel);

		for (int num = 0; num < params->param_outnum["a"]; num++)
		for (int k = 0; k < params->param_dim["a"]; k++)
		{
			is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
			_data[num*params->param_dim["a"]+k] = _d;
		}
		res = cudaMemcpy((void*) (params->s_data["a"]), (void*)(&_data[0]) ,
				_data.size() * sizeof(float_t), cudaMemcpyHostToDevice);
		CHECK(res);

		if(use_for_training) {

			_data.resize(output_channel*channel*kernel_size*kernel_size);

			for (int num = 0; num < params->param_outnum["real_w"]; num++)
			for (int k = 0; k < params->param_dim["real_w"]; k++)
			{
				is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
				_data[num* params->param_dim["real_w"]+k] = _d;
			}
			res = cudaMemcpy((void*) (params->s_data["real_w"]), (void*)(&_data[0]) ,
					_data.size() * sizeof(float_t), cudaMemcpyHostToDevice);
			CHECK(res);

		}
	}

	virtual const void setup() override {
		if(train == phrase)
		means = new blob(output_channel,1,1);
		pad_data = new bin_blob(bottoms[0]->num,(input_dim+2*pad)*(input_dim+2*pad)*channel);
	}

	virtual const int caculate_data_space() override {

		assert(bin_bottoms[0]->channel == channel);

		//signed(I)
		assert(bin_bottoms[0]->dim == input_dim);
		//K
		assert(bottoms[0]->dim == output_dim);
		//real I
		assert(bottoms[1]->dim == input_dim);

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

		if (train == this->phrase)
		resigned_weights();
		//pad for convoluation
		img2bitcol(bin_bottoms[0]->bin_data, channel, kernel_size, stride, pad, input_dim, output_dim, storage_data->bin_data["pad_input"]);

		//caculate binaried convolution
		for (int num = 0; num < bottoms[0]->num; num++) {
			BIT_CACU_COUNT_CONV_CPU(storage_data->bin_data["pad_input"][num], params->bin_data["w"], bottoms[0]->data[num], params->data["a"], kernel_size*kernel_size*channel, tops[0]->data[num]);
		}
	}

	//by CPU
	virtual const void backward(layer_param *&v) override
	{

		img2col(bottoms[1]->data, kernel_size, stride, pad, input_dim, output_dim, storage_data->data["col_data"]);

		for (int num = 0; num < bottoms[1]->data.size(); num++) {

			CACU_DECONV_W_BIN_CPU(storage_data->data["col_data"][num], tops[0]->diff[num], kernel_size, input_dim, pad, channel, stride, v->data["real_w"]);
		}

		//CACU_FIX_GRADIENT_CPU(v->data["real_w"],params->data["a"]);

		CACU_RESET_CPU(storage_data->data["col_data"]);

		for(int num = 0; num < bottoms[1]->num; num++)
		{
			CACU_DECONV_DIFF_CPU(params->data["real_w"], tops[0]->diff[num], kernel_size, channel, output_dim, storage_data->data["col_data"][num]);
		}

		CACU_RESET_CPU(storage_data->data["pad_data"]);

		col2img(storage_data->data["col_data"], kernel_size, stride, pad, input_dim, output_dim, storage_data->data["pad_data"]);

		copy_unpadding_data(storage_data->data["pad_data"], input_dim, pad, bin_bottoms[0]->diff);
	}

	virtual const void save(std::ostream& os) override {
		resigned_weights();

		for (int i = 0; i < layer_name.size(); i++)
		{
			os.write((char*)(&layer_name[i]), sizeof(layer_name[i]));
		}

		for (auto ws : params->bin_data["w"]) {
			for (auto w : ws) os.write((char*)(&w), sizeof(w));
		}
		for (auto ws : params->data["a"]) {
			for (auto w : ws) os.write((char*)(&w), sizeof(w));
		}

		if (use_for_training) {

			for (auto ws : params->data["real_w"]) {
				for (auto w : ws) os.write((char*)(&w), sizeof(w));
			}
		}
	}

	virtual const void load(std::ifstream& is) override {

		string _data;
		char _c;
		float_t _d;
		unsigned int _u;

		for (int i = 0; i < layer_name.size(); i++)
		{
			is.read(&_c, 1);
			_data += _c;
		}

		assert(_data == layer_name);

		for (int num = 0; num < params->bin_data["w"].size(); num++)
		for (int k = 0; k < params->bin_data["w"][0].size(); k++)
		{
			is.read(reinterpret_cast<char*>(&_u), sizeof(unsigned int));
			params->bin_data["w"][num][k] = _u;
		}

		for (int num = 0; num < params->data["a"].size(); num++)
		for (int k = 0; k < params->data["a"][0].size(); k++)
		{
			is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
			params->data["a"][num][k] = _d;
		}

		if (use_for_training) {

			for (int num = 0; num < params->data["real_w"].size(); num++)
			for (int k = 0; k < params->data["real_w"][0].size(); k++)
			{
				is.read(reinterpret_cast<char*>(&_d), sizeof(float_t));
				params->data["real_w"][num][k] = _d;
			}
		}
	}

	virtual const void setup() override {

		if(train == phrase)
		means = new blob(output_channel,1,1);
	}

	virtual const int caculate_data_space() override {

		//signed(I)
		assert(bin_bottoms[0]->bin_data[0].size() == channel*input_dim*input_dim);
		//K
		assert(bottoms[0]->data[0].size() == 1 * output_dim*output_dim);
		//real I
		assert(bin_bottoms[0]->bin_data[0].size() == bottoms[1]->data[0].size());

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

		_param_outnum["real_w"] = output_channel;
		_param_dim["real_w"] = kernel_size*kernel_size*channel;

		_param_outnum["a"] = output_channel;
		_param_dim["a"] = 1;

		_bin_param_outnum["w"] = output_channel;
		_bin_param_dim["w"] = kernel_size*kernel_size*channel;

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

		if (train == this->phrase) {

			_param_outnum["col_data"] = BATCH_SIZE;
			_param_dim["col_data"] = output_dim * output_dim * channel
					* kernel_size * kernel_size;

			_param_outnum["pad_data"] = BATCH_SIZE;
			_param_dim["pad_data"] = channel * (input_dim + 2 * pad)
					* (input_dim + 2 * pad);
		}

		_bin_param_outnum["pad_input"] = BATCH_SIZE;
		_bin_param_dim["pad_input"] = channel * kernel_size * kernel_size
				* output_dim * output_dim;

		////////////////////////////////////////

		_pSTORAGE.push_back(_param_outnum);
		_pSTORAGE.push_back(_param_dim);
		_pSTORAGE.push_back(_bin_param_outnum);
		_pSTORAGE.push_back(_bin_param_dim);
	}

	~bin_conv_layer() {
		delete means;
		delete pad_data;
	}

private:

	int block_size;
	blob *means;
	bin_blob *pad_data;

#if GPU_MODE

	void resigned_weights()
	{
		//mean center
//		CACU_MEAN_GPU(params->s_data["real_w"],output_channel,channel*kernel_size*kernel_size,means->s_data);
//
//		CACU_SUB_GPU_D(params->s_data["real_w"],means->s_data,output_channel,channel*kernel_size*kernel_size,params->s_data["real_w"]);

		BIT_CACU_SIGN_W_GPU(params->s_data["real_w"], params->s_bin_data["w"],output_channel, kernel_size*kernel_size*channel);

		CACU_SUM_SIZE_ABS_GPU(params->s_data["real_w"],output_channel,
				kernel_size * kernel_size * channel,kernel_size * kernel_size * channel,1,
				params->s_data["a"]);

		CACU_SCALE_GPU_A(params->data["a"],(float_t) 1.0 / (kernel_size * kernel_size * channel),output_channel,1,
				params->data["a"], 0);
	}

#else

	void resigned_weights() {

		//mean center
		for (int i = 0; i < output_channel; i++) {
			CACU_SUM_SIZE_CPU(params->data["real_w"][i],
					channel * kernel_size * kernel_size, means->data[i]);

			CACU_SCALE_CPU(means->data[i],
					(float_t) (1.0 / channel * kernel_size * kernel_size),
					means->data[i], 0);
		}

		CACU_SUB_CPU(params->data["real_w"], means->data,
				params->data["real_w"]);

		BIT_CACU_SIGN_W(params->data["real_w"], params->bin_data["w"]);

		for (int num = 0; num < output_channel; num++) {

			CACU_SUM_SIZE_ABS_CPU(params->data["real_w"][num],
					kernel_size * kernel_size * channel,
					params->data["a"][num]);

			CACU_SCALE_CPU(params->data["a"][num],
					(float_t) 1.0 / (kernel_size * kernel_size * channel),
					params->data["a"][num], 0);
		}
	}

#endif

	void DELETE_DATA() {

	}

	bool use_for_training = true;

};

}
;
