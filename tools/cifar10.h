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
#include "../utils.h"
#include "../core/matrix.hpp"

#if GPU_MODE
#include <cuda_runtime.h>
#define CHECK(res) if(res!=cudaSuccess){printf("[cuda error  %d]",res);exit(-1);}
#endif

namespace mycnn {

#ifdef _WIN32

std::wstring StringToWString(const std::string &str)
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
	float_t* sp = &data_blob[0];
	for (unsigned int y = 0; y < height; y++)
	for (unsigned int x = 0; x < width; x++)
	{
		bmp->GetPixel(x, y, &color);
		*(sp + (y * height + x) * 3) = ((float_t)color.GetRed() - (float_t)102.9801);
		*(sp + (y * height + x) * 3 + 1) = ((float_t)color.GetGreen() - (float_t)115.9465);
		*(sp + (y * height + x) * 3 + 2) = ((float_t)color.GetBlue() - (float_t)122.7717);
	}
	delete bmp;
	GdiplusShutdown(gdiplustoken);
}

#elif linux

void read_image(const char* filename, vec_t &data_blob) {

	Mat src = imread(filename);
	unsigned int height = src.rows;
	unsigned int width = src.cols;
	float_t* sp = &data_blob[0];
	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			*(sp + (y * height + x) * 3) = ((float_t) src.at<Vec3b>(y, x)[0]
					- 102.9801);
			*(sp + (y * height + x) * 3 + 1) = ((float_t) src.at<Vec3b>(y, x)[1]
					- 115.9465);
			*(sp + (y * height + x) * 3 + 2) = ((float_t) src.at<Vec3b>(y, x)[2]
					- 122.7717);
		}
}
#endif

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
		for(int j = 0; j < length; j ++) {
			*(start_data+start_index + j) = data_blob[i][j];
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

}
;
