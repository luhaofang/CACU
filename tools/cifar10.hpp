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

#ifdef _WIN32
#include <iostream>
#include <string>
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib") 

using namespace Gdiplus;

#elif linux
#include <string>
#include <libpng/png.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
#endif
#include <iostream>
#if GPU_MODE
#include <cuda_runtime.h>
#endif

#include "../mycnn.h"

#include "../model/cifar10/cifar_10_quick.h"
#include "../model/cifar10/cifar_10_myquick_xnor.h"
#include "../model/cifar10/cifar_10_myquick_xnor_leaky.h"
#include "../model/cifar10/cifar_10_myquick_xnor_sigmoid.h"


const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARDataSize = 1024;

const int kCIFARDataCount = 50000;

void readdata_sub_channel(string filename, vector<vec_t> &data_blob,
		vector<vec_t> &labels, vec_t mean) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	float_t *snp;
	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		labels.push_back(vec_t(1, float_t((label_char))));
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			*(snp + j * 3) = (float_t) ((unsigned char) (buffer[j])) - mean[0];
			*(snp + j * 3 + 1) = (float_t) ((unsigned char) (buffer[j
					+ kCIFARDataSize])) - mean[1];
			*(snp + j * 3 + 2) = (float_t) ((unsigned char) (buffer[j
					+ 2 * kCIFARDataSize])) - mean[2];
		}
		data_blob.push_back(datas);
	}
}

void readdata_sub_dim(string filename, vector<vec_t> &data_blob,
		vector<vec_t> &labels, vec_t &mean) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	float_t *snp;

	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		labels.push_back(vec_t(1, float_t((label_char))));
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			*(snp + j * 3) = (float_t) ((unsigned char) (buffer[j]))
					- mean[j * 3];
			*(snp + j * 3 + 1) =
					(float_t) ((unsigned char) (buffer[kCIFARDataSize + j]))
							- mean[j * 3 + 1];
			*(snp + j * 3 + 2) =
					(float_t) ((unsigned char) (buffer[kCIFARDataSize * 2 + j]))
							- mean[j * 3 + 2];
			//printf("%.5f - %.5f = %.5f\n", (float_t)((unsigned char)(buffer[j])) ,mean[j] , *(snp + j));
		}
		data_blob.push_back(datas);
	}
}

void readdata(string filename, vector<vec_t> &data_blob,
		vector<vec_t> &labels) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	float_t *snp;
	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		labels.push_back(vec_t(1, float_t((label_char))));
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			*(snp + j * 3) = (float_t) ((unsigned char) (buffer[j]));
			*(snp + j * 3 + 1) = (float_t) ((unsigned char) (buffer[j
					+ kCIFARDataSize]));
			*(snp + j * 3 + 2) = (float_t) ((unsigned char) (buffer[j
					+ 2 * kCIFARDataSize]));
		}
		data_blob.push_back(datas);
	}
}

void readdata(string filename, vector<vec_t> &data_blob) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	float_t *snp;
	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			*(snp + j * 3) = (float_t) ((unsigned char) (buffer[j]));
			*(snp + j * 3 + 1) = (float_t) ((unsigned char) (buffer[j
					+ kCIFARDataSize]));
			*(snp + j * 3 + 2) = (float_t) ((unsigned char) (buffer[j
					+ 2 * kCIFARDataSize]));
		}
		data_blob.push_back(datas);
	}
}

#if GPU_MODE

void getdata(unsigned int count, unsigned int start, vector<vec_t> &data_blob,
		float_t *&out_data) {

	cudaError_t res;

	start = start % data_blob.size();

	int length = data_blob[0].size();

	vec_t h_data(count * length);

	float_t *start_data = &h_data[0];

	int start_index;

	for (unsigned int i = start, c = 0; c < count; c++, i++) {

		if (i >= data_blob.size())
			i = 0;
		start_index = c * length;
		for (int j = 0; j < length; j++) {
			*(start_data + start_index + j) = data_blob[i][j];
		}
	}

	res = cudaMemcpy((void*) (out_data), (void*) (start_data),
			count * length * sizeof(float_t), cudaMemcpyHostToDevice);
	CHECK(res);

	vec_t().swap(h_data);
}

#else

void getdata(unsigned int count, unsigned int start, vector<vec_t> &data_blob,
		vector<vec_t> &out_data) {
	float_t *snp, *sdp;

	start = start % data_blob.size();

	for (unsigned int i = start, c = 0; c < count; c++, i++) {
		if (i >= data_blob.size())
		i = 0;
		sdp = &data_blob[i][0];
		snp = &out_data[c][0];
		for (unsigned int j = 0; j < data_blob[0].size(); j++)
		*(snp + j) = *(sdp + j);
	}
}

#endif

vec_t calculate_mean_channel(string &filepath, int filecount) {

	vector<vec_t> mean_data;
	vec_t mean(3);

	//calculate mean
	for (int i = 1; i <= filecount; i++) {
		ostringstream oss;
		oss << filepath << "data_batch_" << i << ".bin";
		readdata((oss.str()), mean_data);
	}

	float_t length = (float_t) mean_data.size() * (mean_data[0].size() / 3);
	float_t r = 0, g = 0, b = 0;
	for (unsigned int i = 0; i < mean_data.size(); i++) {
		for (unsigned int j = 0; j < mean_data[i].size(); j++) {
			if (j % 3 == 0)
				r += (mean_data[i][j] / length);
			else if (j % 3 == 1)
				g += (mean_data[i][j] / length);
			else if (j % 3 == 2)
				b += (mean_data[i][j] / length);
		}
	}

	mean[0] = r;
	mean[1] = g;
	mean[2] = b;

	return mean;
}

vec_t calculate_mean_dim(string &filepath, int filecount) {

	vector<vec_t> mean_data;
	vec_t mean(kCIFARImageNBytes);

	//calculate mean
	for (int i = 1; i <= filecount; i++) {
		ostringstream oss;
		oss << filepath << "data_batch_" << i << ".bin";
		readdata((oss.str()), mean_data);
	}

	float_t length = (float_t) mean_data.size();

	for (unsigned int i = 0; i < mean_data.size(); i++) {
		for (unsigned int j = 0; j < kCIFARImageNBytes; j++) {
			mean[j] += mean_data[i][j];
		}
	}

	for (unsigned int i = 0; i < kCIFARImageNBytes; i++) {
		mean[i] /= length;
	}

	return mean;
}

vec_t calculate_mean_channel() {
	string filepath = "E:/mywork/data/cifar-10-batches-bin/";
	vec_t mean = calculate_mean_dim(filepath, 5);
	printf("%f,%f,%f", mean[0], mean[1], mean[2]);
	return mean;
}

#if GPU_MODE

void train_test() {

	//network *net = cifar_quick();
	network *net = cifar_myquick_xnor_leaky();
	//network *net = cifar_myquick_xnor();
	//network *net = cifar_myquick_xnor_sigmoid();
	//net->load("/home/seal/dataset/experiment/cifar10/test_myquick_5000_xnor_leaky.model");

	vector<vec_t> input_data;
	vector<vec_t> labels;
	vector<vec_t> test_data;
	vector<vec_t> test_labels;

	string location = "/home/seal/dataset/caffe/data/cifar10/";
	string cifar_location = "/home/seal/dataset/caffe/data/cifar10/";

	sgd s(net);
	s.caculate_sgd_data_space();

	vec_t mean = calculate_mean_dim(cifar_location, 5);

	for (int i = 1; i <= 5; i++) {
		ostringstream oss;
		oss << cifar_location << "data_batch_" << i << ".bin";
		readdata_sub_dim((oss.str()), input_data, labels, mean);
	}

	readdata_sub_dim("/home/seal/dataset/caffe/data/cifar10/test_batch.bin",
			test_data, test_labels, mean);

	for (unsigned int i = 1; i <= CACU_MAX_ITER; i++) {

		//int index = 0;
		//vec_t image_data;
		if (i % TEST_ITER == 0) {
			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_data,
					net->net_[net->layers[0]]->bottoms[0]->s_data);

			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_labels,
					net->net_["softmax"]->bottoms[1]->s_data);

			net->predict();

			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, input_data,
					net->net_[net->layers[0]]->bottoms[0]->s_data);

			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, labels,
					net->net_["softmax"]->bottoms[1]->s_data);

			net->predict();

			//s.train(i);

		}

		getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, input_data,
				net->net_[net->layers[0]]->bottoms[0]->s_data);

		getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, labels,
				net->net_["softmax"]->bottoms[1]->s_data);

		s.train(i);

		if (i % SNAPSHOT == 0) {
			ostringstream oss;
			oss << "/home/seal/dataset/experiment/cifar10/test_myquick_" << i
					<< ".model";
			net->save(oss.str().c_str());
		}
	}



}

#else

	void train_test()
	{

		//network *net = cifar_quick();
		network *net = cifar_myquick_xnor();

		//net->load("/home/seal/dataset/experiment/cifar10/test_myquick_bin_10.model");

		blob *input_data = new blob();
		blob *labels = new blob();
		blob *test_data = new blob();
		blob *test_labels = new blob();

		string location = "/home/seal/dataset/caffe/data/cifar10/";
		string cifar_location = "/home/seal/dataset/caffe/data/cifar10/";

		sgd s(net);
		s.caculate_sgd_data_space();

		vec_t mean = calculate_mean_dim(cifar_location, 5);

		for (int i = 1; i <= 5; i++)
		{
			ostringstream oss;
			oss << cifar_location << "data_batch_" << i << ".bin";
			readdata_sub_dim((oss.str()), input_data->data, labels->data, mean);
		}

		readdata_sub_dim("/home/seal/dataset/caffe/data/cifar10/test_batch.bin", test_data->data, test_labels->data, mean);

		for (unsigned int i = 1; i <= CACU_MAX_ITER; i++) {

			//int index = 0;
			//vec_t image_data;
			if (i%TEST_ITER == 0)
			{
				getdata(BATCH_SIZE, (i / TEST_ITER)*BATCH_SIZE, test_data->data, net->net_[net->layers[0]]->bottoms[0]->data);

				getdata(BATCH_SIZE, (i / TEST_ITER)*BATCH_SIZE, test_labels->data, net->net_["softmax"]->bottoms[1]->data);

				net->predict();
			}

			getdata(BATCH_SIZE, (i - 1)*BATCH_SIZE, input_data->data, net->net_[net->layers[0]]->bottoms[0]->data);

			getdata(BATCH_SIZE, (i - 1)*BATCH_SIZE, labels->data, net->net_["softmax"]->bottoms[1]->data);

			s.train(i);

			if (i%SNAPSHOT == 0) {
				ostringstream oss;
				oss << "/home/seal/dataset/experiment/test_myquick_bin_" << i << ".model";
				net->save(oss.str().c_str());
			}
		}

		delete input_data;
		delete labels;
		delete test_data;
		delete test_labels;

	}

	void test_data()
	{
		//network net = resnet18();
		//network *net = cifar_quick(test);
		//network *net = cifar_myquick_xnor(test);
		network *net = cifar_myquick_xnor_leaky(test);
		net->load("/home/seal/dataset/experiment/cifar10/test_myquick_5000.model");

		blob *input_data = new blob();
		blob *labels = new blob();
		blob *test_data = new blob();
		blob *test_labels = new blob();

		//		if (abs(out_data[data_row][data_col]) > 1)
		string cifar_location = "/home/seal/dataset/caffe/data/cifar10/";

		vec_t mean = calculate_mean_dim(cifar_location,5);

		blob *result;

		readdata_sub_dim("/home/seal/dataset/caffe/data/cifar10/test_batch.bin", test_data->data, test_labels->data, mean);

		mycnn::float_t count = 0;
		mycnn::float_t index = 0;
		mycnn::float_t max_ = 0;

		for (unsigned int i = 0; i < 10000 / BATCH_SIZE; i++) {

			getdata(BATCH_SIZE, i *BATCH_SIZE, test_data->data, net->net_[net->layers[0]]->bottoms[0]->data);

			getdata(BATCH_SIZE, i *BATCH_SIZE, test_labels->data, net->net_["softmax"]->bottoms[1]->data);

			result = net->predict();

			for (unsigned int num = 0; num < result->data.size(); num++)
			{
				max_ = result->data[num][0];
				index = 0;
				for (unsigned int i = 0; i < result->data[num].size(); i++) {
					if (result->data[num][i] > max_) {
						max_ = result->data[num][i];
						index = mycnn::float_t(i);
					}
				}
				if (index == net->net_["softmax"]->bottoms[1]->data[num][0])
				count += 1.0;
			}

			printf("test iter %d : %f\n", i, count);
		}

		printf("==============================\n");
		printf("test accuracy: %.10f\n", count / 10000);
		printf("==============================\n");

		delete input_data;
		delete labels;
		delete test_data;
		delete test_labels;

	}

#endif

