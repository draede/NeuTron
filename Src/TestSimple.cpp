
#include "TestSimple.hpp"
#include "NeuTron/Platform/FNN.h"
#include "NeuTron/FNN/Helper.hpp"
#include "NeuTron/FNN/GenericComputingEngine.hpp"
#include "CX/Print.hpp"


using namespace CX;


//example taken from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
int TestSimple()
{
	n_value                in[] = { N_VAL(0.05), N_VAL(0.10) };
	n_value                out[] = { N_VAL(0.01), N_VAL(0.99) };

	NeuTron::FNN::Helper::LayerDef         layers[] =
	{
		{ 2, N_ACTIVATION_IDENTITY, 0,                 N_VAL(0.0) },
		{ 2, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0) },
		{ 2, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0) },
	};
	n_uint32                               cLayers = sizeof(layers) / sizeof(layers[0]);
	n_fnn                                  nnff;
	NeuTron::FNN::GenericComputingEngine   ce;
	n_fnn_layers_info                      *pLayers;
	n_fnn_links_info                       *pLinks;
	n_value                                *vals;
	n_value                                fTmpError;
	n_value                                fInitialError;
	n_value                                fFinalError;
	Status                                 status;

	if ((status = NeuTron::FNN::Helper::Create(&nnff, cLayers, layers, N_VAL(0.5), N_ERROR_SQUARE)))
	{
		pLayers = (n_fnn_layers_info *)nnff.layers.info.pBuffer;
		pLinks  = (n_fnn_links_info *)nnff.links.info.pBuffer;

		//nError = NeuTron::FNN::Helper::InitializeRandomWeights(&nnff);

		vals = (n_value *)((n_uint8 *)nnff.links.data.pBuffer + pLinks->links[0].cbActualOffset);
		vals[0] = N_VAL(0.15); vals[1] = N_VAL(0.25);
		vals[2] = N_VAL(0.20); vals[3] = N_VAL(0.30);

		vals = (n_value *)((n_uint8 *)nnff.layers.data.pBuffer + pLayers->layers[1].cbBiasesOffset);
		vals[0] = N_VAL(0.35); vals[1] = N_VAL(0.35);

		vals = (n_value *)((n_uint8 *)nnff.links.data.pBuffer + pLinks->links[1].cbActualOffset);
		vals[0] = N_VAL(0.40); vals[1] = N_VAL(0.50);
		vals[2] = N_VAL(0.45); vals[3] = N_VAL(0.55);

		vals = (n_value *)((n_uint8 *)nnff.layers.data.pBuffer + pLayers->layers[2].cbBiasesOffset);
		vals[0] = N_VAL(0.60); vals[1] = N_VAL(0.60);

		//Print(stdout, "Initial:\n\n");
		//nError = NeuTron::FNN::Helper::Print(&nnff);

		status = NeuTron::FNN::Helper::Init(&nnff, &ce);

		for (int i = 0; i < 10000; i++)
		{
			status = NeuTron::FNN::Helper::Train(&nnff, 1, in, 2, out, 2, &ce, &fTmpError, &fFinalError);
			if (0 == i)
			{
				fInitialError = fTmpError;
			}
		}

		status = NeuTron::FNN::Helper::Print(&nnff);
		Print(stdout, "\nInitial error : {1}\n", fInitialError);
		Print(stdout, "Final error   : {1}\n", fFinalError);

		status = NeuTron::FNN::Helper::Uninit(&nnff, &ce);

		status = NeuTron::FNN::Helper::Destroy(&nnff);
	}

	return 0;
}
