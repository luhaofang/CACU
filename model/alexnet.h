#include <time.h>
#include "../mycnn.h"

using namespace mycnn;
using namespace boost;

network* alexnet_xnor(type phrase = train) {

	//num,channel,dim
	blob *input_data = new blob(BATCH_SIZE, 3, 227, phrase);
	blob *labels = new blob(BATCH_SIZE, 1, 1);

	static network net;

	bool use_global_stats = true;

	net.phrase = phrase;

	conv_layer *bc1 = new conv_layer("bc1", 227,   //input_dim
			3,	   //channel
			96,	   //output_channel
			11,	   //kernel_size
			4,	   //stride
			2, phrase, 0.01, 0.01);	   //pad
	bc1->bottoms << input_data;
	bc1->set_params_init_value("w", msra);
	bc1->set_params_init_value("bias", constant);
	net << bc1;

	batch_normalization_layer *bn1 = new batch_normalization_layer("bn1", 56, //input_dim
			96, phrase);   //channel
	bn1->bottoms << bc1->tops[0];
	bn1->set_params_init_value("scale", constant, 1.0);
	bn1->set_params_init_value("shift", constant, 0.0);
	bn1->use_global_stats = use_global_stats;
	net << bn1;

	relu_layer *relu1 = new relu_layer("relu1", 56, 96, phrase);
	relu1->bottoms << bn1->tops[0];
	relu1->tops << bn1->tops[0];
	//relu1->slope = 0.01;
	net << relu1;

	max_pooling_layer *mp1 = new max_pooling_layer("pool1", 56,  //input_dim
			96,   //channel
			3,	  //kernel_size
			2, phrase);	  //stride
	mp1->bottoms << relu1->tops[0];
	net << mp1;

	batch_normalization_layer *bn_1a = new batch_normalization_layer("bn_1a",
			28,    //input_dim
			96, phrase);   //channel
	bn_1a->bottoms << mp1->tops[0];
	bn_1a->set_params_init_value("scale", constant, 1.0);
	bn_1a->set_params_init_value("shift", constant, 0.0);
	net << bn_1a;

	bin_activation_layer *ba_1a = new bin_activation_layer("ba_1a", 28, //input_dim
			96,	   //channel
			5,	   //kernel_size
			1,	   //stride
			2, phrase);	   //pad
	ba_1a->bottoms << bn_1a->tops[0];
	net << ba_1a;

	bin_conv_layer *bc_1a = new bin_conv_layer("bc_1a", 28,    //input_dim
			96,	   //channel
			256,   //output_channel
			5,	   //kernel_size
			1,	   //stride
			2, phrase);	   //pad
	bc_1a->bin_bottoms << ba_1a->bin_tops[0];	//for signed(I)
	bc_1a->bottoms << ba_1a->tops[0];	//for K
	bc_1a->bottoms << ba_1a->bottoms[0];	//for real data
	bc_1a->set_params_init_value("real_w", msra);
	net << bc_1a;

	max_pooling_layer *mp2 = new max_pooling_layer("pool2", 28,  //input_dim
			256,   //channel
			3,	  //kernel_size
			2, phrase);	  //stride
	mp2->bottoms << bc_1a->tops[0];
	net << mp2;

	batch_normalization_layer *bn_1b = new batch_normalization_layer("bn_1b",
			14,    //input_dim
			256, phrase);   //channel
	bn_1b->bottoms << mp2->tops[0];
	bn_1b->set_params_init_value("scale", constant, 1.0);
	bn_1b->set_params_init_value("shift", constant, 0.0);
	net << bn_1b;

	bin_activation_layer *ba_1b = new bin_activation_layer("ba_1b", 14, //input_dim
			256,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_1b->bottoms << bn_1b->tops[0];
	net << ba_1b;

	bin_conv_layer *bc_1b = new bin_conv_layer("bc_1b", 14,    //input_dim
			256,	   //channel
			384,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	bc_1b->bin_bottoms << ba_1b->bin_tops[0];	//for signed(I)
	bc_1b->bottoms << ba_1b->tops[0];	//for K
	bc_1b->bottoms << ba_1b->bottoms[0];	//for real data
	bc_1b->set_params_init_value("real_w", msra);
	net << bc_1b;

	batch_normalization_layer *bn_1c = new batch_normalization_layer("bn_1c",
			14,    //input_dim
			384, phrase);   //channel
	bn_1c->bottoms << bc_1b->tops[0];
	bn_1c->set_params_init_value("scale", constant, 1.0);
	bn_1c->set_params_init_value("shift", constant, 0.0);
	net << bn_1c;

	bin_activation_layer *ba_1c = new bin_activation_layer("ba_1c", 14, //input_dim
			384,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_1c->bottoms << bn_1c->tops[0];
	net << ba_1c;

	bin_conv_layer *bc_1c = new bin_conv_layer("bc_1c", 14,    //input_dim
			384,	   //channel
			384,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	bc_1c->bin_bottoms << ba_1c->bin_tops[0];	//for signed(I)
	bc_1c->bottoms << ba_1c->tops[0];	//for K
	bc_1c->bottoms << ba_1c->bottoms[0];	//for real data
	bc_1c->set_params_init_value("real_w", msra);
	net << bc_1c;

	batch_normalization_layer *bn_1d = new batch_normalization_layer("bn_1d",
			14,    //input_dim
			384, phrase);   //channel
	bn_1d->bottoms << bc_1c->tops[0];
	bn_1d->set_params_init_value("scale", constant, 1.0);
	bn_1d->set_params_init_value("shift", constant, 0.0);
	net << bn_1d;

	bin_activation_layer *ba_1d = new bin_activation_layer("ba_1d", 14, //input_dim
			384,	   //channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	ba_1d->bottoms << bn_1d->tops[0];
	net << ba_1d;

	bin_conv_layer *bc_1d = new bin_conv_layer("bc_1d", 14,    //input_dim
			384,	   //channel
			256,   //output_channel
			3,	   //kernel_size
			1,	   //stride
			1, phrase);	   //pad
	bc_1d->bin_bottoms << ba_1d->bin_tops[0];	//for signed(I)
	bc_1d->bottoms << ba_1d->tops[0];	//for K
	bc_1d->bottoms << ba_1d->bottoms[0];	//for real data
	bc_1d->set_params_init_value("real_w", msra);
	net << bc_1d;

	max_pooling_layer *mp3 = new max_pooling_layer("pool3", 14,  //input_dim
			256,   //channel
			3,	  //kernel_size
			2, phrase);	  //stride
	mp3->bottoms << bc_1d->tops[0];
	net << mp3;

	batch_normalization_layer *bn_fc1 = new batch_normalization_layer("bn_fc1",
			7,    //input_dim
			256, phrase);   //channel
	bn_fc1->bottoms << mp3->tops[0];
	bn_fc1->set_params_init_value("scale", constant, 1.0);
	bn_fc1->set_params_init_value("shift", constant, 0.0);
	net << bn_fc1;

	bin_activation_layer *ba_fc1 = new bin_activation_layer("ba_fc1", 7, //input_dim
			256,	   //channel
			7,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	ba_fc1->bottoms << bn_fc1->tops[0];
	net << ba_fc1;

	bin_conv_layer *bc_fc1 = new bin_conv_layer("bc_fc1", 7,    //input_dim
			256,	   //channel
			4096,   //output_channel
			7,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	bc_fc1->bin_bottoms << ba_fc1->bin_tops[0];	//for signed(I)
	bc_fc1->bottoms << ba_fc1->tops[0];	//for K
	bc_fc1->bottoms << ba_fc1->bottoms[0];	//for real data
	bc_fc1->set_params_init_value("real_w", msra);
	net << bc_fc1;

	batch_normalization_layer *bn_fc2 = new batch_normalization_layer("bn_fc2",
			1,    //input_dim
			4096, phrase);   //channel
	bn_fc2->bottoms << bc_fc1->tops[0];
	bn_fc2->set_params_init_value("scale", constant, 1.0);
	bn_fc2->set_params_init_value("shift", constant, 0.0);
	net << bn_fc2;

	bin_activation_layer *ba_fc2 = new bin_activation_layer("ba_fc2", 1, //input_dim
			4096,	   //channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	ba_fc2->bottoms << bn_fc2->tops[0];
	net << ba_fc2;

	bin_conv_layer *bc_fc2 = new bin_conv_layer("bc_fc2", 1,    //input_dim
			4096,	   //channel
			4096,   //output_channel
			1,	   //kernel_size
			1,	   //stride
			0, phrase);	   //pad
	bc_fc2->bin_bottoms << ba_fc2->bin_tops[0];	//for signed(I)
	bc_fc2->bottoms << ba_fc2->tops[0];	//for K
	bc_fc2->bottoms << ba_fc2->bottoms[0];	//for real data
	bc_fc2->set_params_init_value("real_w", msra);
	net << bc_fc2;

//	inner_product_layer *ip1 = new inner_product_layer("ip1", 7,    //input_dim
//			256,   //channel
//			1000, //output_channel
//			phrase);
//	ip1->bottoms << mp2->tops[0];
//	ip1->set_params_init_value("w", msra);
//	ip1->set_params_init_value("bias", constant);
//	net << ip1;
//

	batch_normalization_layer *bn_fc = new batch_normalization_layer("bn_fc", 1, //input_dim
			4096, phrase);   //channel
	bn_fc->bottoms << bc_fc1->tops[0];
	bn_fc->set_params_init_value("scale", constant, 1.0);
	bn_fc->set_params_init_value("shift", constant, 0.0);
	net << bn_fc;

	relu_layer *relu2 = new relu_layer("relu2", 1, 4096, phrase);
	relu2->bottoms << bn_fc->tops[0];
	relu2->tops << bn_fc->tops[0];
	//relu1->slope = 0.01;
	net << relu2;

	inner_product_layer *ip = new inner_product_layer("ip", 1,    //input_dim
			4096,   //channel
			1000, //output_channel
			phrase);
	ip->bottoms << relu2->tops[0];
	ip->set_params_init_value("w", msra);
	ip->set_params_init_value("bias", constant);
	net << ip;

	accuracy_layer *accuracy = new accuracy_layer("accuracy", 1000,  //input_dim
			phrase);  //output_dim
	accuracy->bottoms << ip->tops[0];
	accuracy->bottoms << labels;
	net << accuracy;

	softmax_layer *softmax = new softmax_layer("softmax", 1000,   //input_dim
			phrase);  //output_dim
	softmax->bottoms << ip->tops[0];
	softmax->bottoms << labels;
	net << softmax;

	net.alloc_network_space();

	printf("space costs : %d mb\n",
			(BATCH_SIZE * 3 * 224 * 224 + net.caculate_data_space())
					* sizeof(mycnn::float_t) / 1024 / 1024);

	return &net;
}
