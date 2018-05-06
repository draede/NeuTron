
#pragma once


#include "NeuTron/Platform/FNN.h"
#include "NeuTron/FNN/IComputingEngine.hpp"
#include "CX/Types.hpp"
#include "CX/Status.hpp"
#include "CX/IO/IInputStream.hpp"
#include "CX/IO/IOutputStream.hpp"


namespace NeuTron
{

namespace FNN
{

class Helper
{
public:

	static const CX::UInt32   FLOAT_MAGIC  = 0x464E4E46; //FNNF
	static const CX::UInt32   DOUBLE_MAGIC = 0x444E4E46; //FNNF

	typedef struct _LayerDef
	{
		n_neurons_count         cNeurons;
		n_activation_function   nActivationFunction;
		n_uint32                nFlags;
		n_value                 fBiasValue;
		n_uint8                 cActivationArgs;
		n_value                 activationArgs[N_MAX_ACTIVATION_ARGS_COUNT];
	}LayerDef;

	static CX::Status ComputeMemSize(n_layers_count cLayers, const LayerDef *layers, n_fnn_mem_size *pSize);

	static CX::Status Create(n_fnn *pNNFF, n_layers_count cLayers, const LayerDef *layers, n_value fLearningRate,
	                         n_error_function nErrorFunction, n_uint8 cErrorArgs = 0, n_value *errorArgs = NULL);

	static CX::Status Destroy(n_fnn *pNNFF);

	static CX::Status InitializeRandomWeights(n_fnn *pNNFF, n_value fMin = N_VAL(-1.0), n_value fMax = N_VAL(1.0));

	static CX::Status Save(n_fnn *pNNFF, const CX::Char *szPath);

	static CX::Status Load(n_fnn *pNNFF, const CX::Char *szPath);

	static CX::Status Print(const n_fnn *pNNFF);

	static CX::Status PrintDetailed(const n_fnn *pNNFF);

	static CX::Status Init(n_fnn *pNNFF, IComputingEngine *pCE);

	static CX::Status Uninit(n_fnn *pNNFF, IComputingEngine *pCE);

	static CX::Status Train(n_fnn *pNNFF, n_size cBatchSize,
	                        n_value *inputs, n_neurons_count cInputs, 
	                        n_value *outputs, n_neurons_count cOutputs, 
	                        IComputingEngine *pCE, n_value *pfInitialError, n_value *pfFinalError);

	static CX::Status Test(n_fnn *pNNFF, n_size cBatchSize, n_value *inputs, n_neurons_count cInputs, 
	                       n_value *expected_outputs, n_neurons_count cExpectedOutputs, IComputingEngine *pCE, 
	                       n_value *pfAccuracy);

	static CX::Status Compute(n_fnn *pNNFF, n_value *inputs, n_neurons_count cInputs, n_value *outputs,
	                          n_neurons_count cOutputs, IComputingEngine *pCE);

	static CX::Bool Match(n_value *actual_outputs, n_value *expected_outputs, n_neurons_count cOutputs);

private:

	enum NiceSize
	{
		NiceSize_Bytes,
		NiceSize_KiloBytes,
		NiceSize_MegaBytes,
		NiceSize_GigaBytes,
		NiceSize_TeraBytes,
	};

	Helper();

	~Helper();

	static const CX::Char *GetActivationFunctionName(n_activation_function nActivationFunction);

	static const CX::Char *GetErrorFunctionName(n_error_function nErrorFunction);

	static CX::Status WriteBuffer(n_buffer *pBuffer, CX::IO::IOutputStream *pOS);

	static CX::Status ReadBuffer(n_buffer *pBuffer, CX::IO::IInputStream *pIS);

	static void GetNiceSize(CX::UInt64 cbSize, CX::Double *plfSize, NiceSize *pnNiceSize, CX::String *psNiceSize = NULL);

};

}//namespace FNN

}//namespace NeuTron
