
#include "NeuTron/FNN/Helper.hpp"
#include "CX/Mem.hpp"
#include "CX/Print.hpp"
#include "CX/Util/RndGen.hpp"
#include "CX/IO/FileInputStream.hpp"
#include "CX/IO/FileOutputStream.hpp"
#include <string.h>
#include <windows.h>


using namespace CX;


namespace NeuTron
{

namespace FNN
{

Helper::Helper()
{

}

Helper::~Helper()
{

}

Status Helper::ComputeMemSize(n_layers_count cLayers, const LayerDef *layers, n_fnn_mem_size *pSize)
{
	n_layers_count   cIndex;

	if (N_MIN_LAYERS_COUNT > cLayers)
	{
		return Status_InvalidArg;
	}
	if (N_MAX_LAYERS_COUNT < cLayers)
	{
		return Status_InvalidArg;
	}
	pSize->cbInfo        = sizeof(n_fnn_info);
	pSize->cbLayersInfo  = sizeof(n_fnn_layers_info) + sizeof(n_fnn_layer_info) * (cLayers - 1);
	pSize->cbLayersData  = 0;
	pSize->cbLinksInfo   = sizeof(n_fnn_links_info) + sizeof(n_fnn_link_info) * (cLayers - 2);
	pSize->cbLinksData   = 0;
	pSize->cbInputsInfo  = sizeof(n_fnn_inputs_info);
	pSize->cbOutputsInfo = sizeof(n_fnn_outputs_info);
	for (cIndex = 0; cIndex < cLayers; cIndex++)
	{
		pSize->cbLayersData += 6 * sizeof(n_value) * layers[cIndex].cNeurons;
		if (0 < cIndex)
		{
			pSize->cbLinksData += 2 * sizeof(n_value) * layers[cIndex - 1].cNeurons * layers[cIndex].cNeurons;
		}
	}
	pSize->cbTotal = pSize->cbInfo + 
	                 pSize->cbLayersInfo + pSize->cbLayersData + 
	                 pSize->cbLinksInfo + pSize->cbLinksData + 
	                 pSize->cbInputsInfo + pSize->cbOutputsInfo;

	return Status();
}

Status Helper::Create(n_fnn *pNNFF, n_layers_count cLayers, const LayerDef *layers, n_value fLearningRate,
                      n_error_function nErrorFunction, n_uint8 cErrorArgs/* = 0*/, n_value *errorArgs/* = NULL*/)
{
	n_fnn_mem_size       memsize;
	n_fnn_info           *pInfo;
	n_fnn_layers_info    *pLayersInfo;
	n_fnn_links_info     *pLinksInfo;
	n_fnn_inputs_info    *pInputsInfo;
	n_fnn_outputs_info   *pOutputsInfo;
	n_layers_count       cIndex;
	n_offset             cbOffset;
	Status               status;

	if (!(status = Helper::ComputeMemSize(cLayers, layers, &memsize)))
	{
		return status;
	}

	if (cErrorArgs > N_MAX_ERROR_ARGS_COUNT)
	{
		return Status_InvalidArg;
	}
	if (0 < cErrorArgs && NULL == errorArgs)
	{
		return Status_InvalidArg;
	}

	pNNFF->info.pBuffer         = NULL;
	pNNFF->info.cbSize          = 0;
	pNNFF->layers.info.pBuffer  = NULL;
	pNNFF->layers.info.cbSize   = 0;
	pNNFF->layers.data.pBuffer  = NULL;
	pNNFF->layers.data.cbSize   = 0;
	pNNFF->links.info.pBuffer   = NULL;
	pNNFF->links.info.cbSize    = 0;
	pNNFF->links.data.pBuffer   = NULL;
	pNNFF->links.data.cbSize    = 0;
	pNNFF->inputs.info.pBuffer  = NULL;
	pNNFF->inputs.info.cbSize   = 0;
	pNNFF->inputs.data.pBuffer  = NULL;
	pNNFF->inputs.data.cbSize   = 0;
	pNNFF->outputs.info.pBuffer = NULL;
	pNNFF->outputs.info.cbSize  = 0;
	pNNFF->outputs.data.pBuffer = NULL;
	pNNFF->outputs.data.cbSize  = 0;

	for (;;)
	{
		if (NULL == (pNNFF->info.pBuffer = Mem::Alloc((size_t)memsize.cbInfo)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		pNNFF->info.cbSize = memsize.cbInfo;
		if (NULL == (pNNFF->layers.info.pBuffer = Mem::Alloc((size_t)memsize.cbLayersInfo)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		pNNFF->layers.info.cbSize = memsize.cbLayersInfo;
		if (NULL == (pNNFF->layers.data.pBuffer = Mem::Alloc((size_t)memsize.cbLayersData)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		pNNFF->layers.data.cbSize = memsize.cbLayersData;
		if (NULL == (pNNFF->links.info.pBuffer = Mem::Alloc((size_t)memsize.cbLinksInfo)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		pNNFF->links.info.cbSize = memsize.cbLinksInfo;
		if (NULL == (pNNFF->links.data.pBuffer = Mem::Alloc((size_t)memsize.cbLinksData)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		pNNFF->links.data.cbSize = memsize.cbLinksData;
		if (NULL == (pNNFF->inputs.info.pBuffer = Mem::Alloc((size_t)memsize.cbInputsInfo)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		pNNFF->inputs.info.cbSize = memsize.cbInputsInfo;
		if (NULL == (pNNFF->outputs.info.pBuffer = Mem::Alloc((size_t)memsize.cbOutputsInfo)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		pNNFF->outputs.info.cbSize = memsize.cbOutputsInfo;

		pInfo                 = (n_fnn_info *)pNNFF->info.pBuffer;
		pInfo->cLayers        = cLayers;
		pInfo->cInputs        = layers[0].cNeurons;
		pInfo->cOutputs       = layers[cLayers - 1].cNeurons;
		pInfo->fLearningRate  = fLearningRate;
		pInfo->nErrorFunction = nErrorFunction;
		pInfo->cErrorArgs     = cErrorArgs;
		if (0 < cErrorArgs)
		{
			memcpy(pInfo->errorArgs, errorArgs, cErrorArgs * sizeof(n_value));
		}
		pInfo->memsize        = memsize;

		pLayersInfo           = (n_fnn_layers_info *)pNNFF->layers.info.pBuffer;
		pLayersInfo->cCount   = cLayers;
		cbOffset              = 0;
		for (cIndex = 0; cIndex < cLayers; cIndex++)
		{
			if (layers[cIndex].cActivationArgs > N_MAX_ACTIVATION_ARGS_COUNT)
			{
				status = Status_InvalidArg;

				break;
			}
			if (0 < layers[cIndex].cActivationArgs && NULL == layers[cIndex].activationArgs)
			{
				status = Status_InvalidArg;

				break;
			}

			pLayersInfo->layers[cIndex].cNeurons            = layers[cIndex].cNeurons;
			pLayersInfo->layers[cIndex].nFlags              = layers[cIndex].nFlags;
			pLayersInfo->layers[cIndex].nActivationFunction = layers[cIndex].nActivationFunction;
			pLayersInfo->layers[cIndex].fBiasValue          = layers[cIndex].fBiasValue;
			pLayersInfo->layers[cIndex].cActivationArgs     = layers[cIndex].cActivationArgs;
			if (0 < layers[cIndex].cActivationArgs)
			{
				memcpy(pLayersInfo->layers[cIndex].activationArgs, layers[cIndex].activationArgs, 
				       layers[cIndex].cActivationArgs * sizeof(n_value));
			}
			pLayersInfo->layers[cIndex].cbInOffset          = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons;
			pLayersInfo->layers[cIndex].cbOutOffset         = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons;
			pLayersInfo->layers[cIndex].cbErrorOffset       = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons;
			pLayersInfo->layers[cIndex].cbErrorTmpOffset    = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons;
			pLayersInfo->layers[cIndex].cbBiasesOffset      = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons;
			pLayersInfo->layers[cIndex].cbBiasesTmpOffset   = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons;
		}
		memset(pNNFF->layers.data.pBuffer, 0, (size_t)pNNFF->layers.data.cbSize);

		pLinksInfo            = (n_fnn_links_info *)pNNFF->links.info.pBuffer;
		pLinksInfo->cCount    = cLayers - 1;
		cbOffset              = 0;
		for (cIndex = 0; cIndex < cLayers - 1; cIndex++)
		{
			pLinksInfo->links[cIndex].cRows            = layers[cIndex].cNeurons;
			pLinksInfo->links[cIndex].cColumns         = layers[cIndex + 1].cNeurons;
			pLinksInfo->links[cIndex].cbActualOffset   = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons * layers[cIndex + 1].cNeurons;
			pLinksInfo->links[cIndex].cbAdjustedOffset = cbOffset;
			cbOffset += sizeof(n_value) * layers[cIndex].cNeurons * layers[cIndex + 1].cNeurons;
		}
		memset(pNNFF->links.data.pBuffer, 0, (size_t)pNNFF->links.data.cbSize);

		pInputsInfo              = (n_fnn_inputs_info *)pNNFF->inputs.info.pBuffer;
		pInputsInfo->cBatchSize  = 0;
		pInputsInfo->cValues     = 0;

		pOutputsInfo             = (n_fnn_outputs_info *)pNNFF->outputs.info.pBuffer;
		pOutputsInfo->cBatchSize = 0;
		pOutputsInfo->cValues    = 0;

		break;
	}
	if (!(status))
	{
		Helper::Destroy(pNNFF);
	}

	return status;
}

Status Helper::Destroy(n_fnn *pNNFF)
{
	if (NULL != pNNFF->info.pBuffer)
	{
		Mem::Free(pNNFF->info.pBuffer);
	}
	if (NULL != pNNFF->layers.info.pBuffer)
	{
		Mem::Free(pNNFF->layers.info.pBuffer);
	}
	if (NULL != pNNFF->layers.data.pBuffer)
	{
		Mem::Free(pNNFF->layers.data.pBuffer);
	}
	if (NULL != pNNFF->links.info.pBuffer)
	{
		Mem::Free(pNNFF->links.info.pBuffer);
	}
	if (NULL != pNNFF->links.data.pBuffer)
	{
		Mem::Free(pNNFF->links.data.pBuffer);
	}
	if (NULL != pNNFF->inputs.info.pBuffer)
	{
		Mem::Free(pNNFF->inputs.info.pBuffer);
	}
	if (NULL != pNNFF->outputs.info.pBuffer)
	{
		Mem::Free(pNNFF->outputs.info.pBuffer);
	}
	pNNFF->info.pBuffer         = NULL;
	pNNFF->info.cbSize          = 0;
	pNNFF->layers.info.pBuffer  = NULL;
	pNNFF->layers.info.cbSize   = 0;
	pNNFF->layers.data.pBuffer  = NULL;
	pNNFF->layers.data.cbSize   = 0;
	pNNFF->links.info.pBuffer   = NULL;
	pNNFF->links.info.cbSize    = 0;
	pNNFF->links.data.pBuffer   = NULL;
	pNNFF->links.data.cbSize    = 0;
	pNNFF->inputs.info.pBuffer  = NULL;
	pNNFF->inputs.info.cbSize   = 0;
	pNNFF->inputs.data.pBuffer  = NULL;
	pNNFF->inputs.data.cbSize   = 0;
	pNNFF->outputs.info.pBuffer = NULL;
	pNNFF->outputs.info.cbSize  = 0;
	pNNFF->outputs.data.pBuffer = NULL;
	pNNFF->outputs.data.cbSize  = 0;

	return Status();
}

Status Helper::InitializeRandomWeights(n_fnn *pNNFF, n_value fMin/* = N_VAL(-1.0)*/, n_value fMax/* = N_VAL(1.0)*/)
{
	n_fnn_info          *pInfo;
	n_fnn_layers_info   *pLayersInfo;
	n_fnn_links_info    *pLinksInfo;
	n_value             *values;
	n_layers_count      cLayerIndex;
	n_neurons_count     cRowIndex;
	n_neurons_count     cColumnIndex;
	FILETIME            ftRandTmp;

	pInfo = (n_fnn_info *)pNNFF->info.pBuffer;
	pLayersInfo = (n_fnn_layers_info *)pNNFF->layers.info.pBuffer;
	pLinksInfo = (n_fnn_links_info *)pNNFF->links.info.pBuffer;

	GetSystemTimeAsFileTime(&ftRandTmp);
	Util::RndGen::Get().Seed32(ftRandTmp.dwLowDateTime);
	Util::RndGen::Get().Seed64(((UInt64)ftRandTmp.dwHighDateTime << 32) + ftRandTmp.dwLowDateTime);

	for (cLayerIndex = 0; cLayerIndex < pInfo->cLayers - 1; cLayerIndex++)
	{
		values = (n_value *)((n_uint8 *)pNNFF->links.data.pBuffer +
		                        pLinksInfo->links[cLayerIndex].cbActualOffset);
		for (cRowIndex = 0; cRowIndex < pLinksInfo->links[cLayerIndex].cRows; cRowIndex++)
		{
			for (cColumnIndex = 0; cColumnIndex < pLinksInfo->links[cLayerIndex].cColumns; cColumnIndex++)
			{
				values[cRowIndex * pLinksInfo->links[cLayerIndex].cColumns + cColumnIndex] = 
				                                                  fMin + Util::RndGen::Get().GetFloat() * (fMax - fMin);
			}
		}
		if (N_LAYER_FLAG_BIAS == (N_LAYER_FLAG_BIAS & pLayersInfo->layers[cLayerIndex + 1].nFlags))
		{
			values = (n_value *)((n_uint8 *)pNNFF->layers.data.pBuffer +
			                        pLayersInfo->layers[cLayerIndex + 1].cbBiasesOffset);
			for (n_neurons_count i = 0; i < pLayersInfo->layers[cLayerIndex + 1].cNeurons; i++)
			{
				values[i] = fMin + Util::RndGen::Get().GetFloat() * (fMax - fMin);
			}
		}
	}

	return Status();
}

Status Helper::WriteBuffer(n_buffer *pBuffer, IO::IOutputStream *pOS)
{
	Size     cbSize;
	Status   status;

	if (!(status = pOS->Write(&pBuffer->cbSize, sizeof(pBuffer->cbSize), &cbSize)))
	{
		return status;
	}
	if (sizeof(pBuffer->cbSize) != cbSize)
	{
		return Status_WriteFailed;
	}
	if (!(status = pOS->Write(pBuffer->pBuffer, (Size)pBuffer->cbSize, &cbSize)))
	{
		return status;
	}
	if ((Size)pBuffer->cbSize != cbSize)
	{
		return Status_WriteFailed;
	}

	return Status();
}

Status  Helper::ReadBuffer(n_buffer *pBuffer, CX::IO::IInputStream *pIS)
{
	Size     cbSize;
	Status   status;

	if (!(status = pIS->Read(&pBuffer->cbSize, sizeof(pBuffer->cbSize), &cbSize)))
	{
		return status;
	}
	if (sizeof(pBuffer->cbSize) != cbSize)
	{
		return Status_ReadFailed;
	}
	if (NULL == (pBuffer->pBuffer = Mem::Alloc((Size)pBuffer->cbSize)))
	{
		return Status_MemAllocFailed;
	}
	if (!(status = pIS->Read(pBuffer->pBuffer, (Size)pBuffer->cbSize, &cbSize)))
	{
		return status;
	}
	if ((Size)pBuffer->cbSize != cbSize)
	{
		return Status_ReadFailed;
	}

	return Status();
}

Status Helper::Save(n_fnn *pNNFF, const CX::Char *szPath)
{
	IO::FileOutputStream   os(szPath);
	Size                   cbSize;
	Status                 status;

	if (!os.IsOK())
	{
		return Status_CreateFailed;
	}

	if (sizeof(n_float) == sizeof(n_value))
	{
		if (!(status = os.Write(&FLOAT_MAGIC, sizeof(FLOAT_MAGIC), &cbSize)))
		{
			return status;
		}
	}
	else
	{
		if (!(status = os.Write(&DOUBLE_MAGIC, sizeof(DOUBLE_MAGIC), &cbSize)))
		{
			return status;
		}
	}

	if (!(status = WriteBuffer(&pNNFF->info, &os))) {return status; }
	if (!(status = WriteBuffer(&pNNFF->layers.info, &os))) { return status; }
	if (!(status = WriteBuffer(&pNNFF->layers.data, &os))) { return status; }
	if (!(status = WriteBuffer(&pNNFF->links.info, &os))) { return status; }
	if (!(status = WriteBuffer(&pNNFF->links.data, &os))) { return status; }

	return Status();
}

Status Helper::Load(n_fnn *pNNFF, const CX::Char *szPath)
{
	IO::FileInputStream   is(szPath);
	UInt32                nMagic;
	Size                  cbSize;
	n_fnn_inputs_info     *pInputsInfo;
	n_fnn_outputs_info    *pOutputsInfo;
	Status                status;

	if (!is.IsOK())
	{
		return Status_OpenFailed;
	}
	if (!(status = is.Read(&nMagic, sizeof(nMagic), &cbSize)))
	{
		return status;
	}
	if (sizeof(nMagic) != cbSize)
	{
		return Status_ReadFailed;
	}
	if (sizeof(n_float) == sizeof(n_value))
	{
		if (FLOAT_MAGIC != nMagic)
		{
			return Status_ParseFailed;
		}
	}
	else
	{
		if (DOUBLE_MAGIC != nMagic)
		{
			return Status_ParseFailed;
		}
	}

	pNNFF->layers.pCECTX  = NULL;
	pNNFF->links.pCECTX   = NULL;
	pNNFF->inputs.pCECTX  = NULL;
	pNNFF->outputs.pCECTX = NULL;

	if (!(status = ReadBuffer(&pNNFF->info, &is))) { return status; }
	if (!(status = ReadBuffer(&pNNFF->layers.info, &is))) { return status; }
	if (!(status = ReadBuffer(&pNNFF->layers.data, &is))) { return status; }
	if (!(status = ReadBuffer(&pNNFF->links.info, &is))) { return status; }
	if (!(status = ReadBuffer(&pNNFF->links.data, &is))) { return status; }

	if (NULL == (pNNFF->inputs.info.pBuffer = Mem::Alloc(sizeof(n_fnn_inputs_info))))
	{
		return Status_MemAllocFailed;
	}

	if (NULL == (pNNFF->outputs.info.pBuffer = Mem::Alloc(sizeof(n_fnn_outputs_info))))
	{
		return Status_MemAllocFailed;
	}

	pNNFF->inputs.data.pBuffer  = NULL;
	pNNFF->inputs.data.cbSize   = 0;

	pNNFF->outputs.data.pBuffer = NULL;
	pNNFF->outputs.data.cbSize  = 0;

	pNNFF->inputs.info.cbSize   = sizeof(n_fnn_inputs_info);
	pInputsInfo                 = (n_fnn_inputs_info *)pNNFF->inputs.info.pBuffer;
	pInputsInfo->cBatchSize     = 0;
	pInputsInfo->cValues        = 0;

	pNNFF->outputs.info.cbSize  = sizeof(n_fnn_outputs_info);
	pOutputsInfo                = (n_fnn_outputs_info *)pNNFF->outputs.info.pBuffer;
	pOutputsInfo->cBatchSize    = 0;
	pOutputsInfo->cValues       = 0;

	return Status();
}

const Char *Helper::GetActivationFunctionName(n_activation_function nActivationFunction)
{
	     if (N_ACTIVATION_IDENTITY == nActivationFunction)        { return "identity"; }
	else if (N_ACTIVATION_SIGMOID == nActivationFunction)         { return "sigmoid"; }
	else if (N_ACTIVATION_BINARYSTEP == nActivationFunction)      { return "binary-step"; }
	else if (N_ACTIVATION_TANH == nActivationFunction)            { return "tanh"; }
	else if (N_ACTIVATION_ARCTAN == nActivationFunction)          { return "arctan"; }
	else if (N_ACTIVATION_SOFTSIGN == nActivationFunction)        { return "softsign"; }
	else if (N_ACTIVATION_ISRU == nActivationFunction)            { return "isru"; }
	else if (N_ACTIVATION_RELU == nActivationFunction)            { return "relu"; }
	else if (N_ACTIVATION_LEAKYRELU == nActivationFunction)       { return "leaky-relu"; }
	else if (N_ACTIVATION_PRELU == nActivationFunction)           { return "prelu"; }
	else if (N_ACTIVATION_ELU == nActivationFunction)             { return "elu"; }
	else if (N_ACTIVATION_SELU == nActivationFunction)            { return "selu"; }
	else if (N_ACTIVATION_SRELU == nActivationFunction)           { return "srelu"; }
	else if (N_ACTIVATION_ISRLU == nActivationFunction)           { return "isrlu"; }
	else if (N_ACTIVATION_SOFTPLUS == nActivationFunction)        { return "softplus"; }
	else if (N_ACTIVATION_BENTIDENTITY == nActivationFunction)    { return "bent-identity"; }
	else if (N_ACTIVATION_SOFTEXPONENTIAL == nActivationFunction) { return "soft-exponential"; }
	else if (N_ACTIVATION_SINUSOID == nActivationFunction)        { return "sinusoid"; }
	else if (N_ACTIVATION_SINC == nActivationFunction)            { return "sinc"; }
	else if (N_ACTIVATION_GAUSSIAN == nActivationFunction)        { return "gaussian"; }
	else if (N_ACTIVATION_SOFTMAX == nActivationFunction)         { return "softmax"; }
	else                                                          { return "?!?"; }
}

const Char *Helper::GetErrorFunctionName(n_error_function nErrorFunction)
{
	     if (N_ERROR_SQUARE == nErrorFunction)       { return "square"; }
	else if (N_ERROR_CROSSENTROPY == nErrorFunction) { return "cross-entropy"; }
	else                                             { return "?!?"; }
}

void Helper::GetNiceSize(UInt64 cbSize, Double *plfSize, NiceSize *pnNiceSize, String *psNiceSize/* = NULL*/)
{
	if (1024 > cbSize)
	{
		*plfSize    = (Double)cbSize;
		*pnNiceSize = NiceSize_Bytes;
	}
	else
	if (1048576 > cbSize)
	{
		*plfSize    = cbSize / 1024.0;
		*pnNiceSize = NiceSize_KiloBytes;
	}
	else
	if (1073741824 > cbSize)
	{
		*plfSize    = cbSize / 1048576.0;
		*pnNiceSize = NiceSize_MegaBytes;
	}
	else
	if (1099511627776 > cbSize)
	{
		*plfSize    = cbSize / 1073741824.0;
		*pnNiceSize = NiceSize_GigaBytes;
	}
	else
	{
		*plfSize    = cbSize / 1099511627776.0;
		*pnNiceSize = NiceSize_TeraBytes;
	}
	if (NULL != psNiceSize)
	{
		if (NiceSize_Bytes == *pnNiceSize)
		{
			CX::Print(psNiceSize, "{1} B", cbSize);
		}
		else
		if (NiceSize_KiloBytes == *pnNiceSize)
		{
			CX::Print(psNiceSize, "{1} KB", *plfSize);
		}
		else
		if (NiceSize_MegaBytes == *pnNiceSize)
		{
			CX::Print(psNiceSize, "{1} MB", *plfSize);
		}
		else
		if (NiceSize_GigaBytes == *pnNiceSize)
		{
			CX::Print(psNiceSize, "{1} GB", *plfSize);
		}
		else
		{
			CX::Print(psNiceSize, "{1} TB", *plfSize);
		}
	}
}

Status Helper::Print(const n_fnn *pNNFF)
{
	const n_fnn_info          *pInfo;
	const n_fnn_layers_info   *pLayersInfo;
	const n_fnn_links_info    *pLinksInfo;
	String                    sNiceSize;
	Double                    lfNiceSize;
	NiceSize                  nNiceSize;
	n_uint32                  cLayerIndex;

	pInfo       = (const n_fnn_info *)pNNFF->info.pBuffer;
	pLayersInfo = (const n_fnn_layers_info *)pNNFF->layers.info.pBuffer;
	pLinksInfo  = (const n_fnn_links_info *)pNNFF->links.info.pBuffer;

	GetNiceSize(pInfo->memsize.cbTotal, &lfNiceSize, &nNiceSize, &sNiceSize);

	CX::Print(stdout, "Memory size    : {1}\n", sNiceSize);
	CX::Print(stdout, "Learning rate  : {1}\n", pInfo->fLearningRate);
	CX::Print(stdout, "Error function : {1}\n", GetErrorFunctionName(pInfo->nErrorFunction));
	CX::Print(stdout, "Layers         : {1}\n", pInfo->cLayers);

	for (cLayerIndex = 0; cLayerIndex < pInfo->cLayers; cLayerIndex++)
	{
		if (N_LAYER_FLAG_BIAS == (N_LAYER_FLAG_BIAS & pLayersInfo->layers[cLayerIndex].nFlags))
		{
			CX::Print(stdout, "Layer {1} (neurons = {2}) (activation {3}) (bias {4})\n", cLayerIndex,
			          pLayersInfo->layers[cLayerIndex].cNeurons, 
			          GetActivationFunctionName(pLayersInfo->layers[cLayerIndex].nActivationFunction), 
			          pLayersInfo->layers[cLayerIndex].fBiasValue);
		}
		else
		{
			CX::Print(stdout, "Layer {1} (neurons = {2}) (activation {3})\n", cLayerIndex,
			          pLayersInfo->layers[cLayerIndex].cNeurons, 
			          GetActivationFunctionName(pLayersInfo->layers[cLayerIndex].nActivationFunction));
		}
	}

	return Status();
}

Status Helper::PrintDetailed(const n_fnn *pNNFF)
{
	const n_fnn_info          *pInfo;
	const n_fnn_layers_info   *pLayersInfo;
	const n_fnn_links_info    *pLinksInfo;
	const n_value             *values;
	String                    sNiceSize;
	Double                    lfNiceSize;
	NiceSize                  nNiceSize;
	n_uint32                  cLayerIndex;
	n_uint32                  cNeuronIndex;
	n_uint32                  cRowIndex;
	n_uint32                  cColumnIndex;

	pInfo       = (const n_fnn_info *)pNNFF->info.pBuffer;
	pLayersInfo = (const n_fnn_layers_info *)pNNFF->layers.info.pBuffer;
	pLinksInfo  = (const n_fnn_links_info *)pNNFF->links.info.pBuffer;

	GetNiceSize(pInfo->memsize.cbTotal, &lfNiceSize, &nNiceSize, &sNiceSize);

	CX::Print(stdout, "Memory size    : {1}\n", sNiceSize);
	CX::Print(stdout, "Learning rate  : {1}\n", pInfo->fLearningRate);
	CX::Print(stdout, "Error function : {1}\n", GetErrorFunctionName(pInfo->nErrorFunction));
	CX::Print(stdout, "Layers         : {1}\n", pInfo->cLayers);

	for (cLayerIndex = 0; cLayerIndex < pInfo->cLayers; cLayerIndex++)
	{
		CX::Print(stdout, "Layer {1} (neurons = {2}) (activation {3})\n", cLayerIndex,
		          pLayersInfo->layers[cLayerIndex].cNeurons, 
		          GetActivationFunctionName(pLayersInfo->layers[cLayerIndex].nActivationFunction));

		CX::Print(stdout, "  In:\n");
		values = (const n_value *)((const n_uint8 *)pNNFF->layers.data.pBuffer +
		                              pLayersInfo->layers[cLayerIndex].cbInOffset);
		for (cNeuronIndex = 0; cNeuronIndex < pLayersInfo->layers[cLayerIndex].cNeurons; cNeuronIndex++)
		{
			if (0 == cNeuronIndex)
			{
				CX::Print(stdout, "    {1}", values[cNeuronIndex]);
			}
			else
			{
				CX::Print(stdout, " {1}", values[cNeuronIndex]);
			}
		}
		CX::Print(stdout, "\n");

		CX::Print(stdout, "  Out:\n");
		values = (const n_value *)((const n_uint8 *)pNNFF->layers.data.pBuffer +
		                              pLayersInfo->layers[cLayerIndex].cbOutOffset);
		for (cNeuronIndex = 0; cNeuronIndex < pLayersInfo->layers[cLayerIndex].cNeurons; cNeuronIndex++)
		{
			if (0 == cNeuronIndex)
			{
				CX::Print(stdout, "    {1}", values[cNeuronIndex]);
			}
			else
			{
				CX::Print(stdout, " {1}", values[cNeuronIndex]);
			}
		}
		CX::Print(stdout, "\n");

		CX::Print(stdout, "  Errors:\n");
		values = (const n_value *)((const n_uint8 *)pNNFF->layers.data.pBuffer +
		                              pLayersInfo->layers[cLayerIndex].cbErrorOffset);
		for (cNeuronIndex = 0; cNeuronIndex < pLayersInfo->layers[cLayerIndex].cNeurons; cNeuronIndex++)
		{
			if (0 == cNeuronIndex)
			{
				CX::Print(stdout, "    {1}", values[cNeuronIndex]);
			}
			else
			{
				CX::Print(stdout, " {1}", values[cNeuronIndex]);
			}
		}
		CX::Print(stdout, "\n");

		if (N_LAYER_FLAG_BIAS == (N_LAYER_FLAG_BIAS & pLayersInfo->layers[cLayerIndex].nFlags))
		{
			CX::Print(stdout, "  Bias synapses ({1})\n", pLayersInfo->layers[cLayerIndex].fBiasValue);
			values = (const n_value *)((const n_uint8 *)pNNFF->layers.data.pBuffer +
			                              pLayersInfo->layers[cLayerIndex].cbBiasesOffset);
			for (cNeuronIndex = 0; cNeuronIndex < pLayersInfo->layers[cLayerIndex].cNeurons; cNeuronIndex++)
			{
				if (0 == cNeuronIndex)
				{
					CX::Print(stdout, "    {1}", values[cNeuronIndex]);
				}
				else
				{
					CX::Print(stdout, " {1}", values[cNeuronIndex]);
				}
			}
			CX::Print(stdout, "\n");
		}

		if (cLayerIndex + 1 < pInfo->cLayers)
		{
			CX::Print(stdout, "Link {1} -> {2} (rows = {3}) (columns = {4})\n", cLayerIndex, cLayerIndex + 1,
			          pLinksInfo->links[cLayerIndex].cRows, pLinksInfo->links[cLayerIndex].cColumns);
			CX::Print(stdout, "  Actual:\n");
			values = (const n_value *)((const n_uint8 *)pNNFF->links.data.pBuffer +
			                              pLinksInfo->links[cLayerIndex].cbActualOffset);
			for (cRowIndex = 0; cRowIndex < pLinksInfo->links[cLayerIndex].cRows; cRowIndex++)
			{
				for (cColumnIndex = 0; cColumnIndex < pLinksInfo->links[cLayerIndex].cColumns; cColumnIndex++)
				{
					if (0 == cColumnIndex)
					{
						CX::Print(stdout, "    {1}", 
						          values[cRowIndex * pLinksInfo->links[cLayerIndex].cColumns +cColumnIndex]);
					}
					else
					{
						CX::Print(stdout, " {1}", values[cRowIndex * pLinksInfo->links[cLayerIndex].cColumns + cColumnIndex]);
					}
				}
				CX::Print(stdout, "\n");
			}

			CX::Print(stdout, "  Adjusted:\n");
			values = (const n_value *)((const n_uint8 *)pNNFF->links.data.pBuffer +
			                              pLinksInfo->links[cLayerIndex].cbAdjustedOffset);
			for (cRowIndex = 0; cRowIndex < pLinksInfo->links[cLayerIndex].cRows; cRowIndex++)
			{
				for (cColumnIndex = 0; cColumnIndex < pLinksInfo->links[cLayerIndex].cColumns; cColumnIndex++)
				{
					if (0 == cColumnIndex)
					{
						CX::Print(stdout, "    {1}", 
						          values[cRowIndex * pLinksInfo->links[cLayerIndex].cColumns + cColumnIndex]);
					}
					else
					{
						CX::Print(stdout, " {1}", values[cRowIndex * pLinksInfo->links[cLayerIndex].cColumns + cColumnIndex]);
					}
				}
				CX::Print(stdout, "\n");
			}
		}
	}

	return Status();
}

Status Helper::Init(n_fnn *pNNFF, IComputingEngine *pCE)
{
	return pCE->InitLayersAndLinks(pNNFF);
}

Status Helper::Uninit(n_fnn *pNNFF, IComputingEngine *pCE)
{
	return pCE->UninitLayersAndLinks(pNNFF);
}

Status Helper::Train(n_fnn *pNNFF, n_size cBatchSize,
                     n_value *inputs, n_neurons_count cInputs, 
                     n_value *outputs, n_neurons_count cOutputs, 
                     IComputingEngine *pCE, n_value *pfInitialError, n_value *pfFinalError)
{
	n_fnn_info           *pInfo;
	n_fnn_layers_info    *pLayers;
	n_fnn_links_info     *pLinks;
	n_fnn_inputs_info    *pInputsInfo;
	n_fnn_outputs_info   *pOutputsInfo;
	n_size               cBatchIndex;
	n_layers_count       cLayerIndex;
	Status               status;
	Status               status2;

	pInfo   = (n_fnn_info *)pNNFF->info.pBuffer;
	pLayers = (n_fnn_layers_info *)pNNFF->layers.info.pBuffer;
	pLinks  = (n_fnn_links_info *)pNNFF->links.info.pBuffer;

	if (0 == cBatchSize)
	{
		return Status_InvalidArg;
	}
	if (0 == cInputs)
	{
		return Status_InvalidArg;
	}
	if (cInputs != pInfo->cInputs)
	{
		return Status_InvalidArg;
	}
	if (cOutputs != pInfo->cOutputs)
	{
		return Status_InvalidArg;
	}

	pInputsInfo                 = (n_fnn_inputs_info *)pNNFF->inputs.info.pBuffer;
	pInputsInfo->cBatchSize     = cBatchSize;
	pInputsInfo->cValues        = cInputs;
	pNNFF->inputs.data.pBuffer  = inputs;
	pNNFF->inputs.data.cbSize   = sizeof(n_value) * cBatchSize * cInputs;
	pNNFF->inputs.pCECTX        = NULL;

	pOutputsInfo                = (n_fnn_outputs_info *)pNNFF->outputs.info.pBuffer;
	pOutputsInfo->cBatchSize    = cBatchSize;
	pOutputsInfo->cValues       = cOutputs;
	pNNFF->outputs.data.pBuffer = outputs;
	pNNFF->outputs.data.cbSize  = sizeof(n_value) * cBatchSize * cOutputs;
	pNNFF->outputs.pCECTX       = NULL;

	if (!(status = pCE->InitInputsAndOutputs(pNNFF)))
	{
		return status;
	}

	for (;;)
	{
		for (cBatchIndex = 0; cBatchIndex < cBatchSize; cBatchIndex++)
		{
			//forward

			for (cLayerIndex = 0; cLayerIndex < pLayers->cCount; cLayerIndex++)
			{
				if (0 == cLayerIndex)
				{
					//asign layer 0 in neurons values from the input values
					if (!(status = pCE->Assign(pNNFF, 
					                           pCE->OffsetType_Input, sizeof(n_value) * cInputs * cBatchIndex, 
					                           1, cInputs,
					                           pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset, 
					                           1, pLayers->layers[cLayerIndex].cNeurons)))
					{
						break;
					}
				}
				else
				{
					//calculate layer <cLayerIndex> in neurons values from the synapses input
					if (!(status = pCE->DotProduct(pNNFF, 
					                          pCE->OffsetType_Layer, pLayers->layers[cLayerIndex - 1].cbOutOffset, 
					                          1, pLayers->layers[cLayerIndex - 1].cNeurons, 
					                          pCE->OffsetType_Link, pLinks->links[cLayerIndex - 1].cbActualOffset, 
					                          pLinks->links[cLayerIndex - 1].cRows, pLinks->links[cLayerIndex - 1].cColumns, 
					                          pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset, 
					                          1, pLayers->layers[cLayerIndex].cNeurons)))
					{
						break;
					}
					//add the biases to layer <cLayerIndex> in neurons values
					if (N_LAYER_FLAG_BIAS == (N_LAYER_FLAG_BIAS & pLayers->layers[cLayerIndex].nFlags))
					{
						//multiply bias with bias synapses values
						if (!(status = pCE->Multiply(pNNFF, 
						                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesOffset, 
						                             1, pLayers->layers[cLayerIndex].cNeurons,
						                             pLayers->layers[cLayerIndex].fBiasValue,
						                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset, 
						                             1, pLayers->layers[cLayerIndex].cNeurons)))
						{
							break;
						}
						//add to layer <cLayerIndex> in neurons values
						if (!(status = pCE->Add(pNNFF,
						                        pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset,
						                        1, pLayers->layers[cLayerIndex].cNeurons,
						                        pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset,
						                        1, pLayers->layers[cLayerIndex].cNeurons,
						                        pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset,
						                        1, pLayers->layers[cLayerIndex].cNeurons)))
						{
							break;
						}
					}
				}

				//activate
				if (!(status = pCE->ApplyActivation(pNNFF, pLayers->layers[cLayerIndex].nActivationFunction, 
				                                    pLayers->layers[cLayerIndex].cActivationArgs, 
				                                    pLayers->layers[cLayerIndex].activationArgs, 
				                                    pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset, 
				                                    1, pLayers->layers[cLayerIndex].cNeurons,
				                                    pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbOutOffset, 
				                                    1, pLayers->layers[cLayerIndex].cNeurons)))
				{
					break;
				}
			}
			if (!(status))
			{
				break;
			}

			//backward

			//calculate error
			if (!(status = pCE->ApplyError(pNNFF, pInfo->nErrorFunction, pInfo->cErrorArgs, pInfo->errorArgs, 
			                               pCE->OffsetType_Layer, pLayers->layers[pLayers->cCount - 1].cbOutOffset,
			                               1, pLayers->layers[pLayers->cCount - 1].cNeurons,
			                               pCE->OffsetType_Output, sizeof(n_value) * cOutputs * cBatchIndex,
			                               1, cOutputs, pfFinalError)))
			{
				break;
			}
			if (0 == cBatchIndex)
			{
				*pfInitialError = *pfFinalError;
			}

			//calculate neurons errors
			for (cLayerIndex = pLayers->cCount - 1; cLayerIndex > 0; cLayerIndex--)
			{
				if (pLayers->cCount - 1 == cLayerIndex)
				{
					if (!(status = pCE->ApplyErrorDerivative(pNNFF, pInfo->nErrorFunction, 
					                                     pInfo->cErrorArgs, pInfo->errorArgs, 
					                                     pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbOutOffset, 
					                                     1, pLayers->layers[cLayerIndex].cNeurons, 
					                                     pCE->OffsetType_Output, sizeof(n_value) * cOutputs * cBatchIndex, 
					                                     1, cOutputs, 
					                                     pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorOffset,
					                                     1, pLayers->layers[cLayerIndex].cNeurons)))
					{
						break;
					}
				}
				else
				{
					if (!(status = pCE->DotProduct(pNNFF, 
					                               pCE->OffsetType_Link, pLinks->links[cLayerIndex].cbActualOffset, 
					                               pLinks->links[cLayerIndex].cRows, pLinks->links[cLayerIndex].cColumns, 
					                               pCE->OffsetType_Layer, pLayers->layers[cLayerIndex + 1].cbErrorOffset, 
					                               pLayers->layers[cLayerIndex + 1].cNeurons, 1, 
					                               pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorOffset, 
					                               pLayers->layers[cLayerIndex].cNeurons, 1)))
					{
						break;
					}
				}
				if (!(status = pCE->ApplyActivationDerivative(pNNFF, 
				                                     pLayers->layers[cLayerIndex].nActivationFunction, 
				                                     pLayers->layers[cLayerIndex].cActivationArgs, 
				                                     pLayers->layers[cLayerIndex].activationArgs, 
				                                     pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbOutOffset, 
				                                     1, pLayers->layers[cLayerIndex].cNeurons, 
				                                     pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorTmpOffset, 
				                                     1, pLayers->layers[cLayerIndex].cNeurons)))
				{
					break;
				}
				if (!(status = pCE->Multiply(pNNFF, 
				                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorTmpOffset, 
				                             1, pLayers->layers[cLayerIndex].cNeurons, 
				                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorOffset, 
				                             1, pLayers->layers[cLayerIndex].cNeurons, 
				                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorOffset, 
				                             1, pLayers->layers[cLayerIndex].cNeurons)))
				{
					break;
				}
			}
			if (!status)
			{
				break;
			}

			//calculate adjusted synapses & biases
			for (cLayerIndex = pLayers->cCount - 1; cLayerIndex > 0; cLayerIndex--)
			{
				if (!(status = pCE->OuterProduct(pNNFF, 
				                           pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorOffset, 
				                           1, pLayers->layers[cLayerIndex].cNeurons,
				                           pCE->OffsetType_Layer, pLayers->layers[cLayerIndex - 1].cbOutOffset,
				                           pLayers->layers[cLayerIndex - 1].cNeurons, 1, 
				                           pCE->OffsetType_Link, pLinks->links[cLayerIndex - 1].cbAdjustedOffset, 
				                           pLinks->links[cLayerIndex - 1].cRows, pLinks->links[cLayerIndex - 1].cColumns)))
				{
					break;
				}
				if (!(status = pCE->Multiply(pNNFF, 
				                           pCE->OffsetType_Link, pLinks->links[cLayerIndex - 1].cbAdjustedOffset, 
				                           pLinks->links[cLayerIndex - 1].cRows, pLinks->links[cLayerIndex - 1].cColumns, 
				                           pInfo->fLearningRate, 
				                           pCE->OffsetType_Link, pLinks->links[cLayerIndex - 1].cbAdjustedOffset, 
				                           pLinks->links[cLayerIndex - 1].cRows, pLinks->links[cLayerIndex - 1].cColumns)))
				{
					break;
				}

				//calculate adjusted biases
				if (N_LAYER_FLAG_BIAS == (N_LAYER_FLAG_BIAS & pLayers->layers[cLayerIndex].nFlags))
				{
					if (!(status = pCE->Multiply(pNNFF, 
					                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset, 
					                             1, pLayers->layers[cLayerIndex].cNeurons,
					                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbErrorOffset, 
					                             1, pLayers->layers[cLayerIndex].cNeurons,
					                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset, 
					                             1, pLayers->layers[cLayerIndex].cNeurons)))
					{
						break;
					}
					if (!(status = pCE->Multiply(pNNFF, 
					                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset, 
					                             1, pLayers->layers[cLayerIndex].cNeurons,
					                             pInfo->fLearningRate * pLayers->layers[cLayerIndex].fBiasValue,
					                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset, 
					                             1, pLayers->layers[cLayerIndex].cNeurons)))
					{
						break;
					}
				}
			}
			if (!status)
			{
				break;
			}

			//update synapses & biases
			for (cLayerIndex = 0; cLayerIndex < pLayers->cCount - 1; cLayerIndex++)
			{
				if (!(status = pCE->Substract(pNNFF, 
				                              pCE->OffsetType_Link, pLinks->links[cLayerIndex].cbActualOffset, 
				                              pLinks->links[cLayerIndex].cRows, pLinks->links[cLayerIndex].cColumns, 
				                              pCE->OffsetType_Link, pLinks->links[cLayerIndex].cbAdjustedOffset, 
				                              pLinks->links[cLayerIndex].cRows, pLinks->links[cLayerIndex].cColumns, 
				                              pCE->OffsetType_Link, pLinks->links[cLayerIndex].cbActualOffset, 
				                              pLinks->links[cLayerIndex].cRows, pLinks->links[cLayerIndex].cColumns)))
				{
					break;
				}

				if (N_LAYER_FLAG_BIAS == (N_LAYER_FLAG_BIAS & pLayers->layers[cLayerIndex].nFlags))
				{
					if (!(status = pCE->Substract(pNNFF, 
					                              pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesOffset, 
					                              1, pLayers->layers[cLayerIndex].cNeurons,
					                              pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset, 
					                              1, pLayers->layers[cLayerIndex].cNeurons,
					                              pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesOffset,
					                              1, pLayers->layers[cLayerIndex].cNeurons)))
					{
						break;
					}
				}
			}
			if (!status)
			{
				break;
			}
		}

		if (!status)
		{
			break;
		}

		break;
	}
	status2 = pCE->UninitInputsAndOutputs(pNNFF);
	if (status)
	{
		status = status2;
	}

	return status;
}

Status Helper::Test(n_fnn *pNNFF, n_size cBatchSize, n_value *inputs, n_neurons_count cInputs,
                    n_value *expected_outputs, n_neurons_count cExpectedOutputs, IComputingEngine *pCE, 
                    n_value *pfAccuracy)
{
	n_size    cBatchIndex;
	n_size    cMatched;
	n_value   *outputs;
	Status    status;

	if (NULL == (outputs = (n_value *)Mem::Alloc(sizeof(n_value) * cExpectedOutputs)))
	{
		return Status_MemAllocFailed;
	}
	cMatched = 0;
	for (cBatchIndex = 0; cBatchIndex < cBatchSize; cBatchIndex++)
	{
		if (!(status = Compute(pNNFF, inputs + cBatchIndex * cInputs, cInputs, 
		                       outputs, cExpectedOutputs, pCE)))
		{
			break;
		}
		if (Match(outputs, expected_outputs + cBatchIndex * cExpectedOutputs, cExpectedOutputs))
		{
			cMatched++;
		}
	}
	Mem::Free(outputs);
	*pfAccuracy = (n_value)cMatched * N_VAL(100.0) / (n_value)cBatchSize;

	return status;
}

Status Helper::Compute(n_fnn *pNNFF, n_value *inputs, n_neurons_count cInputs, n_value *outputs,
                       n_neurons_count cOutputs, IComputingEngine *pCE)
{
	n_fnn_info           *pInfo;
	n_fnn_layers_info    *pLayers;
	n_fnn_links_info     *pLinks;
	n_fnn_inputs_info    *pInputsInfo;
	n_layers_count       cLayerIndex;
	Status               status;
	Status               status2;

	pInfo   = (n_fnn_info *)pNNFF->info.pBuffer;
	pLayers = (n_fnn_layers_info *)pNNFF->layers.info.pBuffer;
	pLinks  = (n_fnn_links_info *)pNNFF->links.info.pBuffer;

	if (cInputs != pInfo->cInputs)
	{
		return Status_InvalidArg;
	}
	if (cOutputs != pInfo->cOutputs)
	{
		return Status_InvalidArg;
	}

	pInputsInfo                 = (n_fnn_inputs_info *)pNNFF->inputs.info.pBuffer;
	pInputsInfo->cBatchSize     = 1;
	pInputsInfo->cValues        = cInputs;
	pNNFF->inputs.data.pBuffer  = inputs;
	pNNFF->inputs.data.cbSize   = sizeof(n_value) * cInputs;
	pNNFF->inputs.pCECTX        = NULL;

	if (!(status = pCE->InitInputsAndOutputs(pNNFF)))
	{
		return status;
	}

	for (cLayerIndex = 0; cLayerIndex < pLayers->cCount; cLayerIndex++)
	{
		if (0 == cLayerIndex)
		{
			//asign layer 0 in neurons values from the input values
			if (!(status = pCE->Assign(pNNFF, 
			                           pCE->OffsetType_Input, 0, 
			                           1, cInputs,
			                           pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset, 
			                           1, pLayers->layers[cLayerIndex].cNeurons)))
			{
				break;
			}
		}
		else
		{
			//calculate layer <cLayerIndex> in neurons values from the synapses input
			if (!(status = pCE->DotProduct(pNNFF, 
			                               pCE->OffsetType_Layer, pLayers->layers[cLayerIndex - 1].cbOutOffset, 
			                               1, pLayers->layers[cLayerIndex - 1].cNeurons, 
			                               pCE->OffsetType_Link, pLinks->links[cLayerIndex - 1].cbActualOffset, 
			                               pLinks->links[cLayerIndex - 1].cRows, pLinks->links[cLayerIndex - 1].cColumns, 
			                               pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset, 
			                               1, pLayers->layers[cLayerIndex].cNeurons)))
			{
				break;
			}
			//add the biases to layer <cLayerIndex> in neurons values
			if (N_LAYER_FLAG_BIAS == (N_LAYER_FLAG_BIAS & pLayers->layers[cLayerIndex].nFlags))
			{
				//multiply bias with bias synapses values
				if (!(status = pCE->Multiply(pNNFF, 
				                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesOffset, 
				                             1, pLayers->layers[cLayerIndex].cNeurons,
				                             pLayers->layers[cLayerIndex].fBiasValue,
				                             pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset, 
				                             1, pLayers->layers[cLayerIndex].cNeurons)))
				{
					break;
				}
				//add to layer <cLayerIndex> in neurons values
				if (!(status = pCE->Add(pNNFF,
				                        pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbBiasesTmpOffset,
				                        1, pLayers->layers[cLayerIndex].cNeurons,
				                        pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset,
				                        1, pLayers->layers[cLayerIndex].cNeurons,
				                        pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset,
				                        1, pLayers->layers[cLayerIndex].cNeurons)))
				{
					break;
				}
			}
		}

		//activate
		if (!(status = pCE->ApplyActivation(pNNFF, pLayers->layers[cLayerIndex].nActivationFunction,
		                                    pLayers->layers[cLayerIndex].cActivationArgs, 
		                                    pLayers->layers[cLayerIndex].activationArgs, 
		                                    pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbInOffset, 
		                                    1, pLayers->layers[cLayerIndex].cNeurons,
		                                    pCE->OffsetType_Layer, pLayers->layers[cLayerIndex].cbOutOffset, 
		                                    1, pLayers->layers[cLayerIndex].cNeurons)))
		{
			break;
		}
	}
	if (status)
	{
		//copy here from computing engine!!! (for opencl we need to retrieve it from device memory)
		memcpy(outputs, (n_uint8 *)pNNFF->layers.data.pBuffer + pLayers->layers[pLayers->cCount - 1].cbOutOffset, 
		       cOutputs * sizeof(n_value));
	}
	status2 = pCE->UninitInputsAndOutputs(pNNFF);
	if (status)
	{
		status = status2;
	}

	return status;
}

Bool Helper::Match(n_value *actual_outputs, n_value *expected_outputs, n_neurons_count cOutputs)
{
	size_t    cActualBestIndex   = 0;
	n_value   fActualBestValue   = actual_outputs[0];
	size_t    cExpectedBestIndex = 0;
	n_value   fExpectedBestValue = expected_outputs[0];

	for (size_t i = 1; i < cOutputs; i++)
	{
		if (fActualBestValue < actual_outputs[i])
		{
			cActualBestIndex   = i;
			fActualBestValue   = actual_outputs[i];
		}
		if (fExpectedBestValue < expected_outputs[i])
		{
			cExpectedBestIndex = i;
			fExpectedBestValue = expected_outputs[i];
		}
	}

	return (cActualBestIndex == cExpectedBestIndex);
}

}//namespace FNN

}//namespace NeuTron
