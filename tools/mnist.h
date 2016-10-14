#include <iostream>
#include <fstream>

#include "../utils.h"

#include "../mycnn.h"

#include "../model/mnist/mnist_test_train.h"

using namespace std;

int ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void read_mnist(string filename, vector<vec_t> &vec, float_t scale) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (int i = 0; i < number_of_images; ++i) {
			vec_t tp(n_rows * n_cols);
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));
					tp[r * 28 + c] = (float_t) ((unsigned int) temp) * scale;
				}
			}
			vec.push_back(tp);
		}
	}
}

void read_mnist_label(string filename, vector<vec_t> &vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		for (int i = 0; i < number_of_images; ++i) {
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			vec_t label(1,(float_t) ((unsigned int) temp));
			vec.push_back(label);
		}
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

#if GPU_MODE

void train_test() {

	network *net = mnist_test();

	vector<vec_t> input_data;
	vector<vec_t> labels;
	vector<vec_t> test_data;
	vector<vec_t> test_labels;

	string data_location =
	"/home/seal/dataset/experiment/mnist/train-images.idx3-ubyte";
	string label_location =
	"/home/seal/dataset/experiment/mnist/train-labels.idx1-ubyte";

	string test_data_location =
	"/home/seal/dataset/experiment/mnist/t10k-images.idx3-ubyte";
	string test_label_location =
	"/home/seal/dataset/experiment/mnist/t10k-labels.idx1-ubyte";

	sgd s(net);
	s.caculate_sgd_data_space();

	float_t scale = 0.00390625;

	read_mnist(data_location, input_data, scale);
	read_mnist_label(label_location, labels);

	read_mnist(test_data_location, test_data, scale);
	read_mnist_label(test_label_location, test_labels);

	for (unsigned int i = 1; i <= MAX_ITER; i++) {

		//int index = 0;
		//vec_t image_data;
		if (i % TEST_ITER == 0) {
			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_data,
					net->net_[net->layers[0]]->bottoms[0]->s_data);

			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_labels,
					net->net_["softmax"]->bottoms[1]->s_data);

			net->predict();

		} else {

			getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, input_data,
					net->net_[net->layers[0]]->bottoms[0]->s_data);

			getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, labels,
					net->net_["softmax"]->bottoms[1]->s_data);

			s.train(i);
		}

		if (i % SNAPSHOT == 0) {
			ostringstream oss;
			oss << "/home/seal/dataset/experiment/test_myquick_bin_" << i
			<< ".model";
			net->save(oss.str().c_str());
		}
	}
	vector<vec_t>().swap(input_data);
	vector<vec_t>().swap(labels);
	vector<vec_t>().swap(test_data);
	vector<vec_t>().swap(test_labels);
}

#else

void train_test() {

	network *net = mnist_test();

	blob * input_data = new blob();
	blob * labels = new blob();
	blob * test_data = new blob();
	blob * test_labels = new blob();

	string data_location =
			"/home/seal/dataset/experiment/mnist/train-images.idx3-ubyte";
	string label_location =
			"/home/seal/dataset/experiment/mnist/train-labels.idx1-ubyte";

	string test_data_location =
			"/home/seal/dataset/experiment/mnist/t10k-images.idx3-ubyte";
	string test_label_location =
			"/home/seal/dataset/experiment/mnist/t10k-labels.idx1-ubyte";

	sgd s(net);
	s.caculate_sgd_data_space();

	float_t scale = 0.00390625;

	read_mnist(data_location, input_data->data, scale);
	read_mnist_label(label_location, labels->data);

	read_mnist(test_data_location, test_data->data, scale);
	read_mnist_label(test_label_location, test_labels->data);

	for (unsigned int i = 1; i <= MAX_ITER; i++) {

		//int index = 0;
		//vec_t image_data;
		if (i % TEST_ITER == 0) {
			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_data->data,
					net->net_[net->layers[0]]->bottoms[0]->data);

			getdata(BATCH_SIZE, (i / TEST_ITER) * BATCH_SIZE, test_labels->data,
					net->net_["softmax"]->bottoms[1]->data);

			net->predict();
		}

		else {
			getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, input_data->data,
					net->net_[net->layers[0]]->bottoms[0]->data);

			getdata(BATCH_SIZE, (i - 1) * BATCH_SIZE, labels->data,
					net->net_["softmax"]->bottoms[1]->data);

			s.train(i);
		}

		if (i % SNAPSHOT == 0) {
			ostringstream oss;
			oss << "/home/seal/dataset/experiment/test_myquick_bin_" << i
					<< ".model";
			net->save(oss.str().c_str());
		}
	}

	delete input_data;
	delete labels;
	delete test_data;
	delete test_labels;
}

void test_data() {

	network *net = mnist_test(test);

	net->load("E:/mywork/experiment/test_myquick_5000.model");

	blob *input_data = new blob();
	blob *labels = new blob();
	blob *test_data = new blob();
	blob *test_labels = new blob();

	string test_data_location =
			"/home/seal/dataset/experiment/mnist/t10k-images.idx3-ubyte";
	string test_label_location =
			"/home/seal/dataset/experiment/mnist/t10k-labels.idx1-ubyte";

	float_t scale = 0.00390625;

	blob *result;

	read_mnist(test_data_location, test_data->data, scale);
	read_mnist_label(test_label_location, test_labels->data);

	mycnn::float_t count = 0;
	mycnn::float_t index = 0;
	mycnn::float_t max_ = 0;

	for (unsigned int i = 0; i < 10000 / BATCH_SIZE; i++) {

		getdata(BATCH_SIZE, i * BATCH_SIZE, test_data->data,
				net->net_[net->layers[0]]->bottoms[0]->data);

		getdata(BATCH_SIZE, i * BATCH_SIZE, test_labels->data,
				net->net_["softmax"]->bottoms[1]->data);

		result = net->predict();

		for (unsigned int num = 0; num < result->data.size(); num++) {
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

