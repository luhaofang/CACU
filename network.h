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

	class network {
	public:

		type phrase;
		//layers
		map<char_t, layer*> net_;
		//layer_name
		vector<char_t> layers;

		//using for blob data
		vector<blob*> MEM_SPACE;
		//using for bin_blob data
		vector<bin_blob*> BIN_MEM_SPACE;

		//using for storage data
		map<char_t, layer_param*> P_MEM_STORAGE;

		//using for params
		map<char_t, layer_param*> P_MEM_SPACE;

		void add(layer* layer_) {
			layers.push_back(layer_->layer_name);
			net_[layer_->layer_name] = layer_;
		}

		void alloc_network_space() {
			//alloc calculating space
			ALLOC_DATA_SPACE();
			//alloc storaged data space
			ALLOC_STORAGE_SPACE();
			//alloc parameter data space
			ALLOC_PARAM_DATA_SPACE();
			//alloc temporary data space
			LAYER_SETUP();
		}

#if GPU_MODE

		//caculate net's space
		int caculate_data_space()
		{
			int sum = 0;
			for (int i = 0; i<layers.size(); i++) {
				sum += net_[layers[i]]->caculate_data_space();
			}
			int _sum = 0;
			for (int i = 0; i < MEM_SPACE.size(); i++)
			{
				_sum += MEM_SPACE[i]->num*MEM_SPACE[i]->channel*MEM_SPACE[i]->dim*MEM_SPACE[i]->dim;
				if (phrase == type::train)
					_sum += MEM_SPACE[i]->num*MEM_SPACE[i]->channel*MEM_SPACE[i]->dim*MEM_SPACE[i]->dim;
			}
			for (int i = 0; i < BIN_MEM_SPACE.size(); i++)
			{
				_sum += BIN_MEM_SPACE[i]->num*BIN_MEM_SPACE[i]->channel *BIN_MEM_SPACE[i]->dim*BIN_MEM_SPACE[i]->dim / BIN_SIZE;
				if (phrase == type::train)
					_sum += BIN_MEM_SPACE[i]->num*BIN_MEM_SPACE[i]->channel *BIN_MEM_SPACE[i]->dim*BIN_MEM_SPACE[i]->dim;
			}
			printf("network memcache cost : %d\n", _sum);
			return sum;
		}

#else
		//caculate net's space
		int caculate_data_space() {
			int sum = 0;
			for (int i = 0; i < layers.size(); i++) {
				sum += net_[layers[i]]->caculate_data_space();
			}
			int _sum = 0;
			for (int i = 0; i < MEM_SPACE.size(); i++) {
				_sum += MEM_SPACE[i]->data.size() * MEM_SPACE[i]->data[0].size();
				if (phrase == type::train)
					_sum += MEM_SPACE[i]->diff.size()
					* MEM_SPACE[i]->diff[0].size();
			}
			for (int i = 0; i < BIN_MEM_SPACE.size(); i++) {
				_sum += BIN_MEM_SPACE[i]->bin_data.size()
					* BIN_MEM_SPACE[i]->bin_data[0].size() / BIN_SIZE;
				if (phrase == type::train)
					_sum += BIN_MEM_SPACE[i]->diff.size()
					* BIN_MEM_SPACE[i]->diff[0].size();
			}
			printf("network memcache cost : %d\n", _sum);
			return sum;
		}

#endif

		network& operator <<(layer *layer_) {

			this->add(layer_);
			return *this;
		}

		//reset net's space data for new forwards(&backwards)
		void reset_data_space() {
			for (int i = 0; i < MEM_SPACE.size(); i++) {
				MEM_SPACE[i]->_RESET_DATA();
			}

			for (int i = 0; i < BIN_MEM_SPACE.size(); i++) {
				BIN_MEM_SPACE[i]->_RESET_DATA();
			}
			/*map<char*, layer_param*>::iterator it;
			for (it = P_MEM_STORAGE.begin(); it != P_MEM_STORAGE.end(); ++it)
			{
			it->second->_RESET_DATA();
			}*/
		}

		blob* predict() {
			this->set_phrase(test);
			reset_data_space();
			for (int i = 0; i < this->layers.size(); i++) {
				this->net_[this->layers[i]]->forward();
			}
			return net_[layers[layers.size() - 1]]->tops[0];
		}

		void save(const char* path) {
			std::ofstream os(path, ios::binary);
			os.precision(std::numeric_limits<float_t>::digits10);

			for (int i = 0; i < layers.size(); i++) {
				net_[layers[i]]->save(os);
			}
			os.close();
		}

		void load(const char* path) {
			std::ifstream is(path);
			is.precision(std::numeric_limits<float_t>::digits10);

			for (int i = 0; i < layers.size(); i++) {

				printf("%s\n", net_[layers[i]]->layer_name.c_str());
				net_[layers[i]]->load(is);
			}
			is.close();
		}

		void set_iter(int iter) {
			for (int i = 0; i < layers.size(); i++) {
				net_[layers[i]]->iter = iter;
			}
		}

		void set_phrase(type _phrase) {
			for (int i = 0; i < layers.size(); i++) {
				net_[layers[i]]->phrase = _phrase;
			}
		}

		network() {
			this->phrase = test;
		}

		~network() {

			DEALLOC_SPACE();
		}

		void DEALLOC_SPACE() {
			map<char_t, layer*>::iterator it;
			for (it = net_.begin(); it != net_.end(); ++it) {
				delete it->second;
			}
			map<char_t, layer*>().swap(net_);

			for (int i = 0; i < MEM_SPACE.size(); i++) {
				delete MEM_SPACE[i];
				//MEM_SPACE[i] = NULL;
			}
			vector<blob*>().swap(MEM_SPACE);

			for (int i = 0; i < BIN_MEM_SPACE.size(); i++) {
				delete BIN_MEM_SPACE[i];
				//BIN_MEM_SPACE[i] = NULL;
			}
			vector<bin_blob*>().swap(BIN_MEM_SPACE);

			for (int i = 0; i < layers.size(); i++) {
				delete P_MEM_SPACE[layers[i]];
				//P_MEM_SPACE[layers[i]] = NULL;
				delete P_MEM_STORAGE[layers[i]];
				//P_MEM_STORAGE[layers[i]] = NULL;
			}

			map<char_t, layer_param*>().swap(P_MEM_SPACE);
			map<char_t, layer_param*>().swap(P_MEM_STORAGE);
			vector<char_t>().swap(layers);
		}

	protected:

	private:

		void ALLOC_DATA_SPACE() {
			for (int i = 0; i < layers.size(); i++) {
				char_t layer_name = layers[i];
				P_SPACE _s_params = net_[layer_name]->_PARAMS;
				for (int j = 0; j < _s_params[0].size(); j++) {
					blob* _blob = new blob(BATCH_SIZE, (_s_params)[0][j],
						(_s_params)[1][j], phrase);
					MEM_SPACE.push_back(_blob);
					net_[layer_name]->tops[j] = _blob;
				}
				for (int j = 0; j < _s_params[2].size(); j++) {
					bin_blob* _bin_blob = new bin_blob(BATCH_SIZE,
						(_s_params)[2][j], (_s_params)[3][j], phrase);
					BIN_MEM_SPACE.push_back(_bin_blob);
					net_[layer_name]->bin_tops[j] = _bin_blob;
				}
			}
			for (int i = 0; i < layers.size(); i++) {
				char_t layer_name = layers[i];
				net_[layer_name]->set_probe();
			}
		}

		void ALLOC_STORAGE_SPACE() {
			for (int i = 0; i < layers.size(); i++) {
				char_t layer_name = layers[i];
				pLAYER_SPACE _s_params = net_[layer_name]->_pSTORAGE;

				layer_param* _param = new layer_param(_s_params);
				P_MEM_STORAGE[layer_name] = _param;
				net_[layer_name]->storage_data = _param;

			}
		}

		void ALLOC_PARAM_DATA_SPACE() {
			for (int i = 0; i < layers.size(); i++) {
				char_t layer_name = layers[i];
				pP_SPACE _s_params = net_[layer_name]->_pPARAMS;

				layer_param* _param = new layer_param(_s_params,
					net_[layer_name]->_pPARAMS_TYPE,
					net_[layer_name]->_pPARAMS_VALUE);
				P_MEM_SPACE[layer_name] = _param;
				net_[layer_name]->params = _param;

			}
		}

		void LAYER_SETUP() {
			for (int i = 0; i < layers.size(); i++) {
				char_t layer_name = layers[i];
				net_[layer_name]->setup();
			}
		}

	};

}
;

