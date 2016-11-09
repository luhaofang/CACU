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

#include "../core/math.hpp"
#include "../core/matrix.hpp"
#include "../core/bit_math.hpp"

namespace mycnn {

class layer {
public:

	//basic layer constructor
	layer(char_t name, int input_dim, int channel, int output_channel,
			int kernel_size, int stride, int pad, type phrase, float_t lr_w =
					1.0, float_t lr_b = 1.0) {
	}
	;
	layer(char_t name, int input_dim, int channel, int kernel_size, int stride,
			int pad, type phrase, float_t lr_w = 1.0, float_t lr_b = 1.0) {
	}
	;
	layer(char_t name, int input_dim, int channel, int kernel_size, int stride,
			type phrase, float_t lr_w = 1.0, float_t lr_b = 1.0) {
	}
	;
	layer(char_t name, int input_dim, int channel, int output_channel,
			type phrase, float_t lr_w = 1.0, float_t lr_b = 1.0) {
	}
	;
	layer(char_t name, int input_dim, int output_channel, type phrase,
			float_t lr_w = 1.0, float_t lr_b = 1.0) {
	}
	;
	layer(char_t name, int input_dim, type phrase, float_t lr_w = 1.0,
			float_t lr_b = 1.0) {
	}
	;
	layer(char_t name, type phrase, float_t lr_w = 1.0, float_t lr_b = 1.0) {
	}
	;

	virtual const void forward() = 0;

	virtual const void backward(layer_param *&v) = 0;

	virtual const void load(std::ifstream& is) = 0;

	virtual const void save(std::ostream& os) = 0;

	virtual const int caculate_data_space() = 0;

	virtual const void INIT_SPACE_DATA() = 0;

	virtual const void INIT_PARAM_SAPCE_DATA() = 0;

	virtual const void INIT_STORAGE_DATA() = 0;

	virtual const void setup() = 0;

	virtual ~layer() {
		DEALLOC_SAPCE();
	}

	void set_lr_w(float_t lr) {
		this->lr_w = lr;
	}

	void set_lr_b(float_t lr) {
		this->lr_b = lr;
	}

	void set_probe() {
		for (int i = 0; i < bottoms.size(); i++) {
			if (bottoms[i] != *bottoms.pdata[i]) {
				bottoms[i] = *bottoms.pdata[i];
			}
		}
		for (int i = 0; i < bin_bottoms.size(); i++) {
			if (bin_bottoms[i] != *bin_bottoms.pbin_data[i]) {
				bin_bottoms[i] = *bin_bottoms.pbin_data[i];
			}
		}

		for (int i = 0; i < tops.pdata.size(); i++) {
			if (tops[i] != *tops.pdata[i]) {
				tops[i] = *tops.pdata[i];
			}
		}
	}

	void set_params_init_value(char_t pName, param_init_type init_type,
			float_t value = 0.0) {
		this->_pPARAMS_TYPE[pName] = init_type;
		if (xavier == init_type) {
			this->_pPARAMS_VALUE[pName] =
					sqrt(
							(float_t) 6.0
									/ (kernel_size * kernel_size * channel
											+ kernel_size * kernel_size
													* output_channel));
		} else if (msra == init_type) {
			this->_pPARAMS_VALUE[pName] =
					sqrt(
							(float_t) 4.0
									/ (kernel_size * kernel_size * channel
											+ kernel_size * kernel_size
													* output_channel));
		} else {
			this->_pPARAMS_VALUE[pName] = (float_t) value;
		}
	}

	int iter = 0;

	//feature map output dim
	int output_dim = 0;
	//input feature map channel
	int channel = 0;
	//input dim
	int input_dim = 0;
	//output feature map channel
	int output_channel = 0;
	//kernel size
	int kernel_size = 0;
	//padding size
	int pad = 0;
	//stride size
	int stride = 0;
	//layer's name
	char_t layer_name;

	type phrase;

	//learning rate for w
	float_t lr_w = 1.0;
	//learning rate for b
	float_t lr_b = 1.0;

	layer_param* params = NULL;

	layer_param* storage_data = NULL;

	bin_blobs bin_tops;
	bin_blobs bin_bottoms;

	blobs tops;
	blobs bottoms;

	//need to be deleted
	P_SPACE _PARAMS;
	pP_SPACE _pPARAMS;
	pLAYER_SPACE _pSTORAGE;
	pP_SPACE_INIT_TYPE _pPARAMS_TYPE;
	pP_SPACE_INIT_VALUE _pPARAMS_VALUE;

private:

	void DEALLOC_SAPCE() {

		for (int i = 0; i < bottoms.size(); i++) {
			bottoms[i] = NULL;
		}
		for (int i = 0; i < bin_bottoms.size(); i++) {
			bin_bottoms[i] = NULL;
		}

		for (int i = 0; i < tops.size(); i++) {
			tops[i] = NULL;
		}

		for (int i = 0; i < bin_tops.size(); i++) {
			bin_tops[i] = NULL;
		}

		params = NULL;
		storage_data = NULL;

		for (int i = 0; i < _PARAMS.size(); i++) {
			vector<int>().swap(_PARAMS[i]);
		}
		P_SPACE().swap(_PARAMS);

		for (int i = 0; i < _pPARAMS.size(); i++) {
			map<char_t, int>().swap(_pPARAMS[i]);
		}
		pP_SPACE().swap(_pPARAMS);

		for (int i = 0; i < _pSTORAGE.size(); i++) {
			map<char_t, int>().swap(_pSTORAGE[i]);
		}
		pLAYER_SPACE().swap(_pSTORAGE);

		pP_SPACE_INIT_TYPE().swap(_pPARAMS_TYPE);

		pP_SPACE_INIT_VALUE().swap(_pPARAMS_VALUE);

	}

};

}
;
