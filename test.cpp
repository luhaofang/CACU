#include <time.h>
#include "mycnn.h"
#include <stdlib.h>

using namespace mycnn;
using namespace boost;

#include "model/resnet18.h"
#include "model/cifar_10_quick.h"
#include "model/cifar_10_myquick_xnor.h"

#if GPU_MODE

void train_test() {

	network *net = cifar_myquick_xnor();
	//net->load("E:/mywork/experiment/test_myquick_bin.model");

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

}

#else

void train_test()
{

	//network *net = cifar_quick();
	network *net = cifar_myquick_xnor();

//	net->load("E:/mywork/experiment/test_myquick_bin.model");

	blob *input_data = new blob();
	blob *labels = new blob();
	blob *test_data = new blob();
	blob *test_labels = new blob();

	string location = "E:/mywork/data/cifar-10-batches-bin/";
	string cifar_location = "E:/mywork/data/cifar-10-batches-bin/";

	sgd s(net);
	s.caculate_sgd_data_space();

	vec_t mean = calculate_mean_dim(cifar_location, 5);

	for (int i = 1; i <= 5; i++)
	{
		ostringstream oss;
		oss << cifar_location << "data_batch_" << i << ".bin";
		readdata_sub_dim((oss.str()), input_data->data, labels->data, mean);
	}

	readdata_sub_dim("E:/mywork/data/cifar-10-batches-bin/test_batch.bin", test_data->data, test_labels->data, mean);

	for (unsigned int i = 1; i <= MAX_ITER; i++) {

		//int index = 0;
		//vec_t image_data;
		if (i%TEST_ITER == 0)
		{
			getdata(BATCH_SIZE, (i / TEST_ITER)*BATCH_SIZE, test_data->data, net->net_[net->layers[0]]->bottoms[0]->data);

			getdata(BATCH_SIZE, (i / TEST_ITER)*BATCH_SIZE, test_labels->data, net->net_["softmax"]->bottoms[1]->data);

			net->predict();
		}

		else {

			getdata(BATCH_SIZE, (i - 1)*BATCH_SIZE, input_data->data, net->net_[net->layers[0]]->bottoms[0]->data);

			getdata(BATCH_SIZE, (i - 1)*BATCH_SIZE, labels->data, net->net_["softmax"]->bottoms[1]->data);

			s.train(i);
		}

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
	delete net;
}

void test_data()
{
	//network net = resnet18();
	network *net = cifar_quick(test);
	//network net = cifar_myquick_xnor(test);

	net->load("E:/mywork/experiment/test_myquick_5000.model");

	blob *input_data = new blob();
	blob *labels = new blob();
	blob *test_data = new blob();
	blob *test_labels = new blob();

	string cifar_location = "E:/mywork/data/cifar-10-batches-bin/";

	vec_t mean = calculate_mean_dim(cifar_location,5);

	blob *result;

	readdata_sub_dim("E:/mywork/data/cifar-10-batches-bin/test_batch.bin", test_data->data, test_labels->data, mean);

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
	delete net;
}


#endif

int main() {

	//testim2col();
	//calculate_mean_channel();
	train_test();

	//test_data();
}
