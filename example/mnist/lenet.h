#include <time.h>
#include "../../mycnn.h"

using namespace mycnn;

network* mnist_test(type phrase = train) {

	//num,channel,dim
	blob* input_data = new blob(BATCH_SIZE, 1, 28, phrase);
	blob* labels = new blob(BATCH_SIZE, 1, 1);

	static network net;
	net.phrase = phrase;

	conv_layer *cl1 = new conv_layer("cl1", 28,   //input_dim
			1,	   //channel
			20,	   //output_channel
			5,	   //kernel_size
			1,	   //stride
			0, phrase, 1, 2);	   //pad
	cl1->bottoms << input_data;	//for
	cl1->set_params_init_value("w", xavier);
	cl1->set_params_init_value("bias", constant);
	net << cl1;

	max_pooling_layer *mp1 = new max_pooling_layer("pool1", 24,  //input_dim
			20,   //channel
			2,	  //kernel_size
			2, phrase);	  //stride
	mp1->bottoms << cl1->tops[0];
	net << mp1;

	conv_layer *cl2 = new conv_layer("cl2", 12,   //input_dim
			20,	   //channel
			50,	   //output_channel
			5,	   //kernel_size
			1,	   //stride
			0, phrase, 1, 2);	   //pad
	cl2->bottoms << mp1->tops[0];	//for
	cl2->set_params_init_value("w", xavier);
	cl2->set_params_init_value("bias", constant);
	net << cl2;

	max_pooling_layer *mp2 = new max_pooling_layer("pool2", 8,  //input_dim
			50,   //channel
			2,	  //kernel_size
			2, phrase);	  //stride
	mp2->bottoms << cl2->tops[0];
	net << mp2;

	inner_product_layer *ip1 = new inner_product_layer("ip1", 4,   //input_dim
			50,  //channel
			500,  //output_channel
			phrase, 1, 2);
	ip1->bottoms << mp2->tops[0];
	ip1->set_params_init_value("w", xavier);
	ip1->set_params_init_value("bias", constant);
	net << ip1;

	relu_layer *relu1 = new relu_layer("relu1", 1, 500, phrase);
	relu1->bottoms << ip1->tops[0];
	relu1->tops << ip1->tops[0];
	net << relu1;

	inner_product_layer *ip2 = new inner_product_layer("ip2", 1,   //input_dim
			500,  //channel
			10,  //output_channel
			phrase, 1, 2);
	ip2->bottoms << ip1->tops[0];
	ip2->set_params_init_value("w", xavier);
	ip2->set_params_init_value("bias", constant);
	net << ip2;

	accuracy_layer *accuarcy = new accuracy_layer("accuarcy", 10, phrase);
	accuarcy->bottoms << ip2->tops[0];
	accuarcy->bottoms << labels;
	net << accuarcy;

	softmax_layer *softmax = new softmax_layer("softmax", 10,   //input_dim
			phrase);
	softmax->bottoms << ip2->tops[0];
	softmax->bottoms << labels;
	net << softmax;

	net.alloc_network_space();
	printf("input data costs : %d\n", BATCH_SIZE * 3 * 224 * 224);
	printf("space costs : %d mb\n",
			(net.caculate_data_space()) * sizeof(float) / 1024 / 1024);

	return &net;
}
