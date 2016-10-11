
#include <time.h>
#include "../mycnn.h"

using namespace mycnn;
using namespace boost;


network vgg11()
{
	type phrase = train;


	//num,channel,dim
	blob *input_data = new blob(BATCH_SIZE, 3, 224);
	blob *labels = new blob(BATCH_SIZE, 1, 1);

	network net;

	net.phrase = phrase;

	batch_normalization_layer *bn1 = new batch_normalization_layer("bn1",
		224,   //input_dim		
		3,phrase);   //channel
	bn1->bottoms << input_data;
	bn1->set_params_init_value("scale", constant, 1.0);
	bn1->set_params_init_value("shift", constant, 0.0);
	net << bn1;

	bin_activation_layer *ba1 = new bin_activation_layer("ba1",
		224,   //input_dim
		3,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba1->bottoms << bn1->tops[0];
	net << ba1;

	bin_conv_layer *bc1 = new bin_conv_layer("bc1",
		224,   //input_dim
		3,	   //channel
		64,	   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc1->bin_bottoms << ba1->bin_tops[0];	//for signed(I)
	bc1->bottoms << ba1->tops[0];	//for K
	bc1->bottoms << ba1->bottoms[0];//for real data
	bc1->set_params_init_value("real_w", xavier);
	net << bc1;

	relu_layer *relu1 = new relu_layer("relu1", phrase);
	relu1->bottoms << bc1->tops[0];
	relu1->tops << bc1->tops[0];
	net << relu1;

	max_pooling_layer *mp1 = new max_pooling_layer("pool1",
		224,  //input_dim
		64,   //channel
		2,	  //kernel_size
		2,phrase);	  //stride
	mp1->bottoms << relu1->tops[0];
	net << mp1;

	batch_normalization_layer *bn2 = new batch_normalization_layer("bn2",
		112,   //input_dim		
		64,phrase);   //channel
	bn2->bottoms << mp1->tops[0];
	bn2->set_params_init_value("scale", constant, 1.0);
	bn2->set_params_init_value("shift", constant, 0.0);
	net << bn2;

	bin_activation_layer *ba2 = new bin_activation_layer("ba2",
		112,   //input_dim
		64,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba2->bottoms << bn2->tops[0];
	net << ba2;

	bin_conv_layer *bc2 = new bin_conv_layer("bc2",
		112,   //input_dim
		64,	   //channel
		128,	   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc2->bin_bottoms << ba2->bin_tops[0];	//for signed(I)
	bc2->bottoms << ba2->tops[0];	//for K
	bc2->bottoms << ba2->bottoms[0];//for real data
	bc2->set_params_init_value("real_w", xavier);
	net << bc2;

	relu_layer *relu2 = new relu_layer("relu2", phrase);
	relu2->bottoms << bc2->tops[0];
	relu2->tops << bc2->tops[0];
	net << relu2;

	max_pooling_layer *mp2 = new max_pooling_layer("pool2",
		112,  //input_dim
		128,   //channel
		2,	  //kernel_size
		2,phrase);	  //stride
	mp2->bottoms << relu2->tops[0];
	net << mp2;

	batch_normalization_layer *bn3_1 = new batch_normalization_layer("bn3_1",
		56,   //input_dim		
		128,phrase);   //channel
	bn3_1->bottoms << mp2->tops[0];
	bn3_1->set_params_init_value("scale", constant, 1.0);
	bn3_1->set_params_init_value("shift", constant, 0.0);
	net << bn3_1;

	bin_activation_layer *ba3_1 = new bin_activation_layer("ba3_1",
		56,   //input_dim
		128,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba3_1->bottoms << bn3_1->tops[0];
	net << ba3_1;

	bin_conv_layer *bc3_1 = new bin_conv_layer("bc3_1",
		56,   //input_dim
		128,	   //channel
		256,	   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc3_1->bin_bottoms << ba3_1->bin_tops[0];	//for signed(I)
	bc3_1->bottoms << ba3_1->tops[0];	//for K
	bc3_1->bottoms << ba3_1->bottoms[0];//for real data
	bc3_1->set_params_init_value("real_w", xavier);
	net << bc3_1;

	relu_layer *relu3_1 = new relu_layer("relu3_1", phrase);
	relu3_1->bottoms << bc3_1->tops[0];
	relu3_1->tops << bc3_1->tops[0];
	net << relu3_1;

	batch_normalization_layer *bn3_2 = new batch_normalization_layer("bn3_2",
		56,   //input_dim		
		256,phrase);   //channel
	bn3_2->bottoms << relu3_1->tops[0];
	bn3_2->set_params_init_value("scale", constant, 1.0);
	bn3_2->set_params_init_value("shift", constant, 0.0);
	net << bn3_2;

	bin_activation_layer *ba3_2 = new bin_activation_layer("ba3_2",
		56,   //input_dim
		256,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba3_2->bottoms << bn3_2->tops[0];
	net << ba3_2;

	bin_conv_layer *bc3_2 = new bin_conv_layer("bc3_2",
		56,   //input_dim
		256,	   //channel
		256,	   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc3_2->bin_bottoms << ba3_2->bin_tops[0];	//for signed(I)
	bc3_2->bottoms << ba3_2->tops[0];	//for K
	bc3_2->bottoms << ba3_2->bottoms[0];//for real data
	bc3_2->set_params_init_value("real_w", xavier);
	net << bc3_2;

	relu_layer *relu3_2 = new relu_layer("relu3_2", phrase);
	relu3_2->bottoms << bc3_2->tops[0];
	relu3_2->tops << bc3_2->tops[0];
	net << relu3_2;

	max_pooling_layer *mp3 = new max_pooling_layer("pool3",
		56,  //input_dim
		256,   //channel
		2,	  //kernel_size
		2,phrase);	  //stride
	mp3->bottoms << relu3_2->tops[0];
	net << mp3;

	batch_normalization_layer *bn4_1 = new batch_normalization_layer("bn4_1",
		28,   //input_dim		
		256,phrase);   //channel
	bn4_1->bottoms << mp3->tops[0];
	bn4_1->set_params_init_value("scale", constant, 1.0);
	bn4_1->set_params_init_value("shift", constant, 0.0);
	net << bn4_1;

	bin_activation_layer *ba4_1 = new bin_activation_layer("ba4_1",
		28,   //input_dim
		256,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba4_1->bottoms << bn4_1->tops[0];
	net << ba4_1;

	bin_conv_layer *bc4_1 = new bin_conv_layer("bc4_1",
		28,   //input_dim
		256,	   //channel
		512,	   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc4_1->bin_bottoms << ba4_1->bin_tops[0];	//for signed(I)
	bc4_1->bottoms << ba4_1->tops[0];	//for K
	bc4_1->bottoms << ba4_1->bottoms[0];//for real data
	bc4_1->set_params_init_value("real_w", xavier);
	net << bc4_1;

	relu_layer *relu4_1 = new relu_layer("relu4_1", phrase);
	relu4_1->bottoms << bc4_1->tops[0];
	relu4_1->tops << bc4_1->tops[0];
	net << relu4_1;

	batch_normalization_layer *bn4_2 = new batch_normalization_layer("bn4_2",
		28,   //input_dim		
		512,phrase);   //channel
	bn4_2->bottoms << relu4_1->tops[0];
	bn4_2->set_params_init_value("scale", constant, 1.0);
	bn4_2->set_params_init_value("shift", constant, 0.0);
	net << bn4_2;

	bin_activation_layer *ba4_2 = new bin_activation_layer("ba4_2",
		28,   //input_dim
		512,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba4_2->bottoms << bn4_2->tops[0];
	net << ba4_2;

	bin_conv_layer *bc4_2 = new bin_conv_layer("bc4_2",
		28,   //input_dim
		512,	   //channel
		512,	   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc4_2->bin_bottoms << ba4_2->bin_tops[0];	//for signed(I)
	bc4_2->bottoms << ba4_2->tops[0];	//for K
	bc4_2->bottoms << ba4_2->bottoms[0];//for real data
	bc4_2->set_params_init_value("real_w", xavier);
	net << bc4_2;

	relu_layer *relu4_2 = new relu_layer("relu4_2", phrase);
	relu4_2->bottoms << bc4_2->tops[0];
	relu4_2->tops << bc4_2->tops[0];
	net << relu4_2;

	max_pooling_layer *mp4 = new max_pooling_layer("pool4",
		28,  //input_dim
		512,   //channel
		2,	  //kernel_size
		2,phrase);	  //stride
	mp4->bottoms << relu4_2->tops[0];
	net << mp4;

	batch_normalization_layer *bn5_1 = new batch_normalization_layer("bn5_1",
		14,   //input_dim		
		512,phrase);   //channel
	bn5_1->bottoms << mp4->tops[0];
	bn5_1->set_params_init_value("scale", constant, 1.0);
	bn5_1->set_params_init_value("shift", constant, 0.0);
	net << bn5_1;

	bin_activation_layer *ba5_1 = new bin_activation_layer("ba5_1",
		14,   //input_dim
		512,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba5_1->bottoms << bn5_1->tops[0];
	net << ba5_1;

	bin_conv_layer *bc5_1 = new bin_conv_layer("bc5_1",
		14,   //input_dim
		512,	   //channel
		512,	   //output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc5_1->bin_bottoms << ba5_1->bin_tops[0];	//for signed(I)
	bc5_1->bottoms << ba5_1->tops[0];	//for K
	bc5_1->bottoms << ba5_1->bottoms[0];//for real data
	bc5_1->set_params_init_value("real_w", xavier);
	net << bc5_1;

	relu_layer *relu5_1 = new relu_layer("relu5_1", phrase);
	relu5_1->bottoms << bc5_1->tops[0];
	relu5_1->tops << bc5_1->tops[0];
	net << relu5_1;

	batch_normalization_layer *bn5_2 = new batch_normalization_layer("bn5_2",
		14,   //input_dim		
		512,phrase);   //channel
	bn5_2->bottoms << relu5_1->tops[0];
	bn5_2->set_params_init_value("scale", constant, 1.0);
	bn5_2->set_params_init_value("shift", constant, 0.0);
	net << bn5_2;

	bin_activation_layer *ba5_2 = new bin_activation_layer("ba5_2",
		14,   //input_dim
		512,	   //channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	ba5_2->bottoms << bn5_2->tops[0];
	net << ba5_2;

	bin_conv_layer *bc5_2 = new bin_conv_layer("bc5_2",
		14,   //input_dim
		512,	//channel
		512,	//output_channel
		3,	   //kernel_size
		1,	   //stride
		1,phrase);	   //pad
	bc5_2->bin_bottoms << ba5_2->bin_tops[0];	//for signed(I)
	bc5_2->bottoms << ba5_2->tops[0];	//for K
	bc5_2->bottoms << ba5_2->bottoms[0];//for real data
	bc5_2->set_params_init_value("real_w", xavier);
	net << bc5_2;

	relu_layer *relu5_2 = new relu_layer("relu5_2", phrase);
	relu5_2->bottoms << bc5_2->tops[0];
	relu5_2->tops << bc5_2->tops[0];
	net << relu5_2;

	max_pooling_layer *mp5 = new max_pooling_layer("pool5",
		14,  //input_dim
		512,   //channel
		2,	  //kernel_size
		2,phrase);	  //stride
	mp5->bottoms << relu5_2->tops[0];
	net << mp5;

	batch_normalization_layer *bnfc_1 = new batch_normalization_layer("bnfc_1",
		7,   //input_dim		
		512,phrase);   //channel
	bnfc_1->bottoms << mp5->tops[0];
	bnfc_1->set_params_init_value("scale", constant, 1.0);
	bnfc_1->set_params_init_value("shift", constant, 0.0);
	net << bnfc_1;

	bin_activation_layer *bafc_1 = new bin_activation_layer("bafc_1",
		7,   //input_dim
		512,	   //channel
		7,	   //kernel_size
		1,	   //stride
		0,phrase);	   //pad
	bafc_1->bottoms << bnfc_1->tops[0];
	net << bafc_1;

	bin_conv_layer *bcfc_1 = new bin_conv_layer("bcfc_1",
		7,   //input_dim
		512,	   //channel
		4096,	   //output_channel
		7,	   //kernel_size
		1,	   //stride
		0,phrase);	   //pad
	bcfc_1->bin_bottoms << bafc_1->bin_tops[0];	//for signed(I)
	bcfc_1->bottoms << bafc_1->tops[0];	//for K
	bcfc_1->bottoms << bafc_1->bottoms[0];//for real data
	bcfc_1->set_params_init_value("real_w", xavier);
	net << bcfc_1;

	relu_layer *relufc_1 = new relu_layer("relufc_1", phrase);
	relufc_1->bottoms << bcfc_1->tops[0];
	relufc_1->tops << bcfc_1->tops[0];
	net << relufc_1;

	batch_normalization_layer *bnfc_2 = new batch_normalization_layer("bnfc_2",
		1,   //input_dim		
		4096,phrase);   //channel
	bnfc_2->bottoms << relufc_1->tops[0];
	bnfc_2->set_params_init_value("scale", constant, 1.0);
	bnfc_2->set_params_init_value("shift", constant, 0.0);
	net << bnfc_2;

	bin_activation_layer *bafc_2 = new bin_activation_layer("bafc_2",
		1,   //input_dim
		4096,	   //channel
		1,	   //kernel_size
		1,	   //stride
		0,phrase);	   //pad
	bafc_2->bottoms << bnfc_2->tops[0];
	net << bafc_2;

	bin_conv_layer *bcfc_2 = new bin_conv_layer("bcfc_2",
		1,   //input_dim
		4096,	   //channel
		1000,	   //output_channel
		1,	   //kernel_size
		1,	   //stride
		0,phrase);	   //pad
	bcfc_2->bin_bottoms << bafc_2->bin_tops[0];	//for signed(I)
	bcfc_2->bottoms << bafc_2->tops[0];	//for K
	bcfc_2->bottoms << bafc_2->bottoms[0];//for real data
	bcfc_2->set_params_init_value("real_w", xavier);
	net << bcfc_2;

	relu_layer *relufc_2 = new relu_layer("relufc_2", phrase);
	relufc_2->bottoms << bcfc_2->tops[0];
	relufc_2->tops << bcfc_2->tops[0];
	net << relufc_2;

	softmax_layer *softmax = new softmax_layer("softmax",
		1000,   //input_dim		
		phrase);  //output_dim
	softmax->bottoms << relufc_2->tops[0];
	softmax->bottoms << labels;
	net << softmax;

	net.alloc_network_space();

	//printf("%s layer's param size : %d\n", net.net_["norm"]->layer_name, bl1.get()->data[0].size());
	printf("space costs : %d mb\n", (BATCH_SIZE * 3 * 224 * 224 + net.caculate_data_space()) * sizeof(float) / 1024 / 1024);

	return net;
}
