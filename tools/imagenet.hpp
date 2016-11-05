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
#include <mycnn::char_t>
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib") 

using namespace Gdiplus;
#endif

#include <libpng/png.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
using namespace std;
using namespace cv;

#if GPU_MODE
#include <cuda_runtime.h>
#endif

#include "../mycnn.h"
#include "../utils.h"

#include "../model/resnet18.h"

using namespace mycnn;

const int ImageNBytes = 3 * 224 * 224;
const int DataSize = 224 * 224;

#ifdef _WIN32

std::wstring StringToWString(const std::mycnn::char_t &str)
{
	std::wstring wstr(str.length(), L' ');
	std::copy(str.begin(), str.end(), wstr.begin());
	return wstr;
}

void read_image(const char* filename, vec_t &data_blob)
{
	GdiplusStartupInput gdiplusstartupinput;
	ULONG_PTR gdiplustoken;
	GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);
	Bitmap* bmp = new Bitmap(StringToWString(filename).c_str());
	unsigned int height = bmp->GetHeight();
	unsigned int width = bmp->GetWidth();
	Color color;
	mycnn::float_t* sp = &data_blob[0];
	for (unsigned int y = 0; y < height; y++)
	for (unsigned int x = 0; x < width; x++)
	{
		bmp->GetPixel(x, y, &color);
		*(sp + (y * height + x) * 3) = ((mycnn::float_t)color.GetRed() - (mycnn::float_t)102.9801);
		*(sp + (y * height + x) * 3 + 1) = ((mycnn::float_t)color.GetGreen() - (mycnn::float_t)115.9465);
		*(sp + (y * height + x) * 3 + 2) = ((mycnn::float_t)color.GetBlue() - (mycnn::float_t)122.7717);
	}
	delete bmp;
	GdiplusShutdown(gdiplustoken);
}

void readimage2vec(mycnn::char_t filepath, mycnn::float_t *&data, vec_t &mean)
{
	GdiplusStartupInput gdiplusstartupinput;
	ULONG_PTR gdiplustoken;
	GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);
	std::ifstream is(filepath);
	mycnn::char_t line;
	int count = 0;
	mycnn::float_t* mp = &mean[0];
	mycnn::float_t* sp = data;
	Color color;
	Bitmap* bmp;

	bmp = new Bitmap(StringToWString(line).c_str());

	for (unsigned int y = 0; y < 224; y++)
	for (unsigned int x = 0; x < 224; x++)
	{
		bmp->GetPixel(x, y, &color);
		*(sp + (y * 224 + x) * 3) = ((mycnn::float_t)color.GetRed() - *(mp + (y * 224 + x) * 3));
		*(sp + (y * 224 + x) * 3 + 1) = ((mycnn::float_t)color.GetGreen() - *(mp + (y * 224 + x) * 3 + 1));
		*(sp + (y * 224 + x) * 3 + 2) = ((mycnn::float_t)color.GetBlue() - *(mp + (y * 224 + x) * 3 + 2));
	}
	delete bmp;

}

void calculate_mean_dim(mycnn::char_t filepath, mycnn::char_t dirpath) {

	vec_t mean(ImageNBytes);

	GdiplusStartupInput gdiplusstartupinput;
	ULONG_PTR gdiplustoken;
	GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);
	std::ifstream is(filepath);
	mycnn::char_t line;
	int count = 0;
	mycnn::float_t* sp = &mean[0];
	Color color;
	Bitmap* bmp;
	while (getline(is, line))
	{
		bmp = new Bitmap(StringToWString(dirpath + line).c_str());

		for (unsigned int y = 0; y < 224; y++)
		for (unsigned int x = 0; x < 224; x++)
		{
			bmp->GetPixel(x, y, &color);
			*(sp + (y * 224 + x) * 3) += ((mycnn::float_t)color.GetRed());
			*(sp + (y * 224 + x) * 3 + 1) += ((mycnn::float_t)color.GetGreen());
			*(sp + (y * 224 + x) * 3 + 2) += ((mycnn::float_t)color.GetBlue());
		}
		delete bmp;
		count += 1;
		if (count % 10000 == 0)
		printf("process %d images!\n", count);
	}
	GdiplusShutdown(gdiplustoken);
	is.close();

	mycnn::float_t length = (mycnn::float_t)count;

	for (unsigned int i = 0; i < ImageNBytes; i++) {
		mean[i] /= length;
	}

	vector<mycnn::char_t> paths = split(filepath, "/");
	mycnn::char_t name = split(paths[paths.size() - 1], ".")[0];
	mycnn::char_t outpath = "";
	for (int i = 0; i < paths.size() - 1; i++)
	outpath += (paths[i] + "/");
	outpath += (name + ".meanfile");
	std::ofstream os(outpath, ios::binary);
	for (int i = 0; i < ImageNBytes; i++)
	os.write((char*)(sp + i), sizeof(*(sp + i)));
	os.close();
}

#endif

void readimage2vec(mycnn::char_t filepath, vec_t &data, vec_t &mean) {

	mycnn::float_t* mp = &mean[0];
	mycnn::float_t* sp = &data[0];
	Mat src = imread((filepath), IMREAD_COLOR);
	unsigned int height = 224;
	unsigned int width = 224;

	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			*(sp + (y * height + x) * 3) =
					((mycnn::float_t) src.at<Vec3b>(y, x)[0]
							- *(mp + (y * height + x) * 3));
			*(sp + (y * height + x) * 3 + 1) = ((mycnn::float_t) src.at<Vec3b>(
					y, x)[1] - *(mp + (y * height + x) * 3 + 1));
			*(sp + (y * height + x) * 3 + 2) = ((mycnn::float_t) src.at<Vec3b>(
					y, x)[2] - *(mp + (y * height + x) * 3 + 2));
		}

}

void calculate_mean_dim(mycnn::char_t filepath, mycnn::char_t dirpath) {

	vec_t mean(ImageNBytes);

	Mat src;
	unsigned int height;
	unsigned int width;
	std::ifstream is(filepath);
	mycnn::char_t line;
	int count = 0;
	mycnn::float_t* sp = &mean[0];

	while (getline(is, line)) {

		src = imread((dirpath + split(line, "\t")[0]), IMREAD_COLOR);
		height = src.rows;
		width = src.cols;

		for (unsigned int y = 0; y < height; y++)
			for (unsigned int x = 0; x < width; x++) {
				*(sp + (y * height + x) * 3) += ((mycnn::float_t) src.at<Vec3b>(
						y, x)[0]);
				*(sp + (y * height + x) * 3 + 1) += ((mycnn::float_t) src.at<
						Vec3b>(y, x)[1]);
				*(sp + (y * height + x) * 3 + 2) += ((mycnn::float_t) src.at<
						Vec3b>(y, x)[2]);
			}
		count += 1;
		if (count % 10000 == 0) {
			printf("process %d images!\n", count);
			//break;
		}
	}
	is.close();

	float_t length = (float_t) count;

	for (unsigned int i = 0; i < ImageNBytes; i++) {
		mean[i] /= length;
	}

	vector<mycnn::char_t> paths = split(filepath, "/");
	mycnn::char_t name = split(paths[paths.size() - 1], ".")[0];
	mycnn::char_t outpath = "/";
	for (int i = 0; i < paths.size() - 1; i++)
		outpath += (paths[i] + "/");
	outpath += (name + ".meanfile");
	std::ofstream os(outpath, ios::binary);
	for (int i = 0; i < ImageNBytes; i++) {
		//printf("%f\n", *(sp + i));
		os.write((char*) (sp + i), sizeof(*(sp + i)));
	}
	os.close();
}

void read_mean(string meanfile, vec_t &mean) {
	std::ifstream is(meanfile);
	mycnn::float_t _f;
	for (int i = 0; i < mean.size(); i++) {
		is.read(reinterpret_cast<char*>(&_f), sizeof(mycnn::float_t));
		mean[i] = _f;
	}
	is.close();
}

#if GPU_MODE

void getdata(unsigned int count, unsigned int start,
		vector<mycnn::char_t> &data_blob, vec_t &mean,
		blob *&out_data) {

	assert(out_data->num == count);
	assert(out_data->channel * out_data->dim * out_data->dim == ImageNBytes);

	cudaError_t res;
	mycnn::char_t dirpath =
				"/home/seal/dataset/imagenet/data/ILSVRC2012/ILSVRC2012_img_train_224/";
	start = start % data_blob.size();

	vec_t h_data(count*ImageNBytes);
	vec_t data(ImageNBytes);

	mycnn::float_t *start_data = &h_data[0];
	mycnn::float_t *sp = &data[0];
	int start_index;

	for (unsigned int i = start, c = 0; c < count; c++, i++) {
		if (i >= data_blob.size())
			i = 0;
		readimage2vec(dirpath+data_blob[i], data, mean);
		start_data = &h_data[0] + c * ImageNBytes;
		for(int j = 0; j < ImageNBytes ; j ++){
			*(start_data + j) = *(sp + j);
		}
	}

	res = cudaMemcpy((void*) (out_data->s_data), (void*) (&h_data[0]),
			h_data.size() * sizeof(mycnn::float_t),
			cudaMemcpyHostToDevice);
	CHECK(res);

	vec_t().swap(h_data);
	vec_t().swap(data);
}

void getdata(unsigned int count, unsigned int start, vector<vec_t> &data_blob,
		blob *&out_data) {
	assert(out_data->num == count);
	cudaError_t res;

	start = start % data_blob.size();

	vec_t h_data(count);

	float_t *start_data = &h_data[0];

	for (unsigned int i = start, c = 0; c < count; c++, i++) {

		if (i >= data_blob.size())
			i = 0;

		*(start_data + c) = data_blob[i][0];

	}

	res = cudaMemcpy((void*) (out_data->s_data), (void*) (start_data),
			count * sizeof(float_t), cudaMemcpyHostToDevice);
	CHECK(res);

	vec_t().swap(h_data);
}

void train_test() {

	network *net = resnet18();
	//net->load("/home/seal/dataset/experiment/cifar10/test_myquick_5000_xnor_leaky.model");

	vector<char_t> input_data;
	vector<vec_t> labels;
	vector<char_t> test_data;
	vector<vec_t> test_labels;

	sgd s(net);
	s.caculate_sgd_data_space();

	vec_t mean(ImageNBytes);
	read_mean("/home/seal/dataset/imagenet/train_list.meanfile", mean);

	std::ifstream is("/home/seal/dataset/imagenet/train_img.txt");
	char_t line;
	float_t label;
	while (getline(is, line)) {
		input_data.push_back(split(line, "\t")[0]);
		label = atof(split(line, "\t")[1].c_str());
		labels.push_back(vec_t(1, label));
	}

	for (unsigned int i = 1; i <= CACU_MAX_ITER; i++) {

		//int index = 0;
		//vec_t image_data;
//		if (i % TEST_ITER == 0) {
//			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_data,
//					net->net_[net->layers[0]]->bottoms[0]->s_data);
//
//			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_labels,
//					net->net_["softmax"]->bottoms[1]->s_data);
//
//			net->predict();
//
//			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, input_data,
//					net->net_[net->layers[0]]->bottoms[0]->s_data);
//
//			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, labels,
//					net->net_["softmax"]->bottoms[1]->s_data);
//
//			net->predict();
//
//			//s.train(i);
//
//		}

		getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, input_data, mean,
				net->net_[net->layers[0]]->bottoms[0]);

		getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, labels,
				net->net_["softmax"]->bottoms[1]);

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

void getdata(unsigned int count, unsigned int start,
		vector<mycnn::char_t> &data_blob, vec_t &mean,
		vector<vec_t> &out_data) {
	mycnn::float_t *snp;
	mycnn::char_t sdp;

	start = start % data_blob.size();

	for (unsigned int i = start, c = 0; c < count; c++, i++) {
		if (i >= data_blob.size())
		i = 0;
		snp = &out_data[c][0];
		sdp = data_blob[i];
		readimage2vec(sdp, snp, mean);
	}
}

void getdatalist(mycnn::char_t filepath, vector<mycnn::char_t> &filelist) {
	std::ifstream is(filepath);
	mycnn::char_t line;
	while (getline(is, line)) {
		filelist.push_back(line);
	}
}

#endif
