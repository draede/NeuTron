
#include "TestMNIST.hpp"
#include "NeuTron/Platform/FNN.h"
#include "NeuTron/FNN/Helper.hpp"
#include "NeuTron/FNN/GenericComputingEngine.hpp"
#include "NeuTron/MNIST/Converter.hpp"
#include "CX/Print.hpp"
#include "CX/Util/Timer.hpp"


using namespace CX;


int TestMNIST()
{
	NeuTron::FNN::Helper::LayerDef         layers[] =
	{
//97.739998 % accuracy after 43 epochs:
/*
		{ 784, N_ACTIVATION_IDENTITY, 0,                 N_VAL(0.0), 0 },
		{ 196, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
		{ 10,  N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
*/

//98.019997 % accuracy after 30 epochs:
		{ 784, N_ACTIVATION_IDENTITY, 0,                 N_VAL(0.0), 0 },
		{ 196, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
		{ 196, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
		{ 10,  N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },

//97.419998 % accuracy after 32 epochs:
/*
		{ 784, N_ACTIVATION_IDENTITY, 0,                 N_VAL(0.0), 0 },
		{ 196, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
		{ 196, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
		{ 196, N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
		{ 10,  N_ACTIVATION_SIGMOID,  N_LAYER_FLAG_BIAS, N_VAL(1.0), 0 },
*/
	};
	n_uint32                               cLayers = sizeof(layers) / sizeof(layers[0]);
	n_fnn                                  nnff;
	NeuTron::FNN::GenericComputingEngine   ce;
	n_value                                fInitialError;
	n_value                                fFinalError;
	NeuTron::DataSet                       datasetTraining;
	NeuTron::DataSet                       datasetTesting;
	CX::Util::Timer                        timer;
	Double                                 lfTrainingElapsed;
	Double                                 lfTestingElapsed;
	n_value                                fAccuracy;
	n_value                                fMaxAccuracy;
	Status                                 status;

	if ((status = NeuTron::FNN::Helper::Create(&nnff, cLayers, layers, N_VAL(0.5), N_ERROR_SQUARE)))
	{
		status = NeuTron::FNN::Helper::InitializeRandomWeights(&nnff);

		status = NeuTron::FNN::Helper::Init(&nnff, &ce);

		status = datasetTraining.Open("mnist_train_inputs.dat", "mnist_train_outputs.dat", 268435456, True); // 256 MB
		status = datasetTesting.Open("mnist_test_inputs.dat", "mnist_test_outputs.dat", 268435456, True); // 256 MB

		size_t cEpochs = 10;

		fMaxAccuracy = N_VAL(0.0);
		for (size_t i = 0; i < cEpochs; i++)
		{
			status = datasetTraining.Reset();
			lfTrainingElapsed = 0.0;
			while ((status = datasetTraining.Next()))
			{
				timer.ResetTimer();
				status = NeuTron::FNN::Helper::Train(&nnff,
				                                     datasetTraining.GetCurrentIterationBatchSize(),
				                                     datasetTraining.GetInputs(), datasetTraining.GetInputsCount(),
				                                     datasetTraining.GetOutputs(), datasetTraining.GetOutputsCount(),
				                                     &ce, &fInitialError, &fFinalError);
				lfTrainingElapsed = timer.GetElapsedTime();
			}

			status = datasetTesting.Reset();
			lfTestingElapsed = 0;
			while ((status = datasetTesting.Next()))
			{
				timer.ResetTimer();
				status = NeuTron::FNN::Helper::Test(&nnff, 
				                                    datasetTesting.GetCurrentIterationBatchSize(),
				                                    datasetTesting.GetInputs(), datasetTesting.GetInputsCount(), 
				                                    datasetTesting.GetOutputs(), datasetTesting.GetOutputsCount(), &ce, 
				                                    &fAccuracy);
				lfTestingElapsed = timer.GetElapsedTime();
				if (fMaxAccuracy < fAccuracy)
				{
					fMaxAccuracy = fAccuracy;
				}
			}

			Print(stdout,
			      "EPOCH {1} => {2}% (max {3}%), error({4} -> {5}), timings({6}s, {7}s)\n",
			      i, fAccuracy, fMaxAccuracy, fInitialError, fFinalError, lfTrainingElapsed, lfTestingElapsed);

			String   sPath;

			Print(&sPath, "mnist_fnn_epoch_{1}_accuracy_{2}.dat", i, fAccuracy);
			status = NeuTron::FNN::Helper::Save(&nnff, sPath.c_str());
		}

		status = NeuTron::FNN::Helper::Uninit(&nnff, &ce);

		status = NeuTron::FNN::Helper::Destroy(&nnff);
	}

	return 0;
}

void ConvertMNIST()
{
	Status   status;

	status = NeuTron::MNIST::Converter::ConvertImages("t10k-images.idx3-ubyte", "mnist_test_inputs.dat");
	status = NeuTron::MNIST::Converter::ConvertLabels("t10k-labels.idx1-ubyte", "mnist_test_outputs.dat");
	status = NeuTron::MNIST::Converter::ConvertImages("train-images.idx3-ubyte", "mnist_train_inputs.dat");
	status = NeuTron::MNIST::Converter::ConvertLabels("train-labels.idx1-ubyte", "mnist_train_outputs.dat");
}
