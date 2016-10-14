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

class max_pooling_layer: public layer {

public:

	max_pooling_layer(char_t layer_name, int input_dim, int channel,
			int kernel_size, int stride, type phrase = train,
			float_t lr_w = 1.0, float_t lr_b = 1.0) :
			layer(layer_name, input_dim, channel, kernel_size, stride, phrase,
					lr_w, lr_b) {
		this->layer_name = layer_name;
		this->kernel_size = kernel_size;
		this->channel = channel;
		this->input_dim = input_dim;
		this->output_channel = channel;
		if ((input_dim - kernel_size) % stride != 0)
			this->pad = kernel_size - stride;
		this->output_dim = (input_dim + pad - kernel_size) / stride + 1;
		this->phrase = phrase;
		this->stride = stride;
		this->set_lr_w(lr_w);
		this->set_lr_b(lr_b);
		INIT_SPACE_DATA();
		INIT_PARAM_SAPCE_DATA();
		INIT_STORAGE_DATA();
	}

#if GPU_MODE

	virtual const void forward() override
	{

		CACU_M_POOLING_GPU(bottoms[0]->data, bottoms[0]->num, kernel_size,input_dim, output_dim, output_dim*output_dim*output_channel, channel,
				stride, tops[0]->data,storage_data->data["index_data"]);
	}

	virtual const void backward(layer_param *&v) override
	{
		//added to activied feature map
		copy2dest_gpu(tops[0]->diff, storage_data->data["index_data"], tops[0]->num,
				output_dim, input_dim, channel, kernel_size, stride,bottoms[0]->diff);
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
		float_t *sdp, *sip, *snp;
		float_t *si_out_cp, *sd_out_cp, *sn_out_cp;
		int sd_out, sn_out,sd_out_c,index;

		for (int num = 0; num < bottoms[0]->data.size(); num++) {
			sdp = &bottoms[0]->data[num][0];
			sip = &storage_data->data["index_data"][num][0];
			snp = &tops[0]->data[num][0];

			for (int i = 0; i < output_dim; i++)
			for (int j = 0; j < output_dim; j++) {
				sn_out = (i * output_dim + j) * channel;
				sd_out = (i * input_dim + j) *stride* channel;
				for (int c = 0; c < channel; c++) {
					si_out_cp = sip + sn_out + c;
					sn_out_cp = snp + sn_out + c;
					sd_out_c = sd_out + c;
					for (int ki = 0; ki < kernel_size && (ki + i*stride) < input_dim; ki++)
					for (int kj = 0; kj < kernel_size && (kj + j*stride) < input_dim; kj++)
					{
						sd_out_cp = sdp + sd_out_c + (ki * input_dim + kj) * channel;
						if (*sn_out_cp < *sd_out_cp || (ki == 0 && kj == 0))
						{
							index = ki*kernel_size + kj;
							*si_out_cp = index;
							*sn_out_cp = *sd_out_cp;
						}
					}
				}
			}
		}
	}

	virtual const void backward(layer_param *&v) override
	{
		//added to activied feature map
		copy2dest(tops[0]->diff, storage_data->data["index_data"], output_dim, input_dim, channel, kernel_size, stride, bottoms[0]->diff);

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

		_param_outnum.push_back(channel);
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

		_param_outnum["index_data"] = BATCH_SIZE;
		_param_dim["index_data"] = output_dim * output_dim * channel;

		////////////////////////////////////////

		_pSTORAGE.push_back(_param_outnum);
		_pSTORAGE.push_back(_param_dim);
		_pSTORAGE.push_back(_bin_param_outnum);
		_pSTORAGE.push_back(_bin_param_dim);
	}

	void DELETE_DATA() {

	}

	~max_pooling_layer() {

		DELETE_DATA();

	}

private:

};

}
;
