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

class accuracy_layer: public layer {

public:

	accuracy_layer(char_t layer_name, int input_dim, type phrase, float_t lr_w =
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

		os.open("/home/seal/dataset/experiment/cifar10/test_accuary_bin.txt");
		os.precision(std::numeric_limits<float_t>::digits10);

	}

#if GPU_MODE

	virtual const void forward() override
	{
		if (test == this->phrase)
		{

			cudaError_t res;

			CACU_SOFTMAX_GPU(bottoms[0]->data, bottoms[0]->num, input_dim, tops[0]->data);

			vec_t predict_data(bottoms[0]->num*input_dim);

			res = cudaMemcpy((void*) (&predict_data[0]), (void*) (tops[0]->s_data),
					bottoms[0]->num * input_dim * sizeof(float_t), cudaMemcpyDeviceToHost);
//			printf("acc_layer:");
//			for(int i =0; i < predict_data.size();i++)
//			printf("%.10f,",predict_data[i]);
//			printf("\n");

			int index = 0;

			int label;

			int size = input_dim;

			float_t loss = 0.0;

			float_t *cuda_loss;
			res = cudaMalloc((void**) (&cuda_loss), sizeof(float_t));
			CHECK(res);

			CACU_CE_LOSS_GPU(tops[0]->data, bottoms[1]->data, bottoms[0]->num, cuda_loss);

			res = cudaMemcpy((void*) (&loss), (void*) (cuda_loss), sizeof(float_t), cudaMemcpyDeviceToHost);
			CHECK(res);

			loss = loss / (float_t)bottoms[0]->num;

			//for test code
			vec_t predict_labels(bottoms[1]->num);

			res = cudaMemcpy((void*) (&predict_labels[0]), (void*) (bottoms[1]->s_data),
					bottoms[1]->num * sizeof(float_t), cudaMemcpyDeviceToHost);
			CHECK(res);
			float_t sum = 0;

			float_t max_ = 0;

			for(int i = 0; i < bottoms[1]->num;i++) {
				max_= 0;
				for (int j = 0; j < size; j++) {
					if (predict_data[i*input_dim + j] > max_) {
						max_ = predict_data[i*input_dim + j];
						index = j;
					}
				}
				if (index == (int)predict_labels[i])
				sum += 1.0;
			}

			printf("==============================\n");
			printf("test accuracy: %f\n", sum / bottoms[0]->num);
			os << sum / bottoms[0]->num << "\n";

			printf("loss:%f\n", loss);
			printf("==============================\n");
		}
	}

	virtual const void backward(layer_param *&v) override
	{

	}

	virtual const void save(std::ostream& os) override {

	}

	virtual const void load(std::ifstream& is) override {

	}

	virtual const void setup() override {

	}

	virtual const int caculate_data_space() override {

		printf("%s top costs : %d \n", layer_name.c_str(), 0);

		printf("%s params costs %d \n", layer_name.c_str(), params->caculate_space());

		return 0;
	}

#else

	virtual const void forward() override
	{
		if (test == this->phrase) {

			blob data(bottoms[0]->data.size(), bottoms[0]->data[0].size(), test);

			int size = bottoms[0]->data[0].size();

			float_t *sp, max_, *p, sum;
			for (int num = 0; num < bottoms[0]->data.size(); num++)
			{
				sp = &data.data[num][0];
				max_ = bottoms[0]->data[num][0];
				p = &bottoms[0]->data[num][0];
				sum = 0.0;
				for (int i = 0; i < size; i++)
				max_ = max(*(p + i), max_);
				for (int i = 0; i < size; i++)
				{
					*(sp + i) = exp(*(p + i) - max_);
					sum += *(sp + i);
				}
				for (int i = 0; i < size; i++)
				{
					*(sp + i) = (*(sp + i) / sum);
				}
			}

			float_t index = 0;

			int label;

			sum = 0.0;
			for (int num = 0; num < data.data.size(); num++)
			{
				max_ = data.data[num][0];
				index = 0;
				for (int i = 0; i < size; i++) {
					if (*(&data.data[num][0] + i) > max_) {
						max_ = *(&data.data[num][0] + i);
						index = (float_t)i;
					}
				}
				if (index == bottoms[1]->data[num][0])
				sum += 1.0;
			}
			printf("==============================\n");
			printf("test accuracy: %.10f\n", sum / bottoms[0]->data.size());
			os << sum / bottoms[0]->data.size() << "\n";

			float_t loss = 0.0;
			for (int num = 0; num < bottoms[0]->data.size(); num++)
			{
				label = (int)bottoms[1]->data[num][0];
				//cout << data.data[num][label] << endl;
				loss -= log(data.data[num][label]);
			}
			loss = loss / (float_t)BATCH_SIZE;
			printf("loss         : %.10f\n", loss);
			printf("==============================\n");

		}
	}

	virtual const void backward(layer_param *&v) override
	{

	}

	virtual const void save(std::ostream& os) override {

	}

	virtual const void load(std::ifstream& is) override {

	}

	virtual const void setup() override {

	}

	virtual const int caculate_data_space() override {

		printf("%s top costs : %d \n", layer_name.c_str(), 0);

		printf("%s params costs %d \n", layer_name.c_str(), params->caculate_space());

		return 0;
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

	~accuracy_layer() {
		os.close();

	}

private:

	std::ofstream os;

};

}
;
