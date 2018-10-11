
#include "NeuTron/FNN/OpenCLComputingEngine.hpp"
#include <math.h>


using namespace CX;


namespace NeuTron
{

namespace FNN
{

OpenCLComputingEngine::OpenCLComputingEngine()
{
}

OpenCLComputingEngine::~OpenCLComputingEngine()
{
}

void *OpenCLComputingEngine::GetPtr(n_fnn *pNNFF, OffsetType nOffsetType, n_offset cbOffset, n_size cbSize)
{
	n_buffer   *pBuffer;

	if (OffsetType_Layer == nOffsetType)
	{
		pBuffer = &pNNFF->layers.data;
	}
	else
	if (OffsetType_Link == nOffsetType)
	{
		pBuffer = &pNNFF->links.data;
	}
	else
	if (OffsetType_Input == nOffsetType)
	{
		pBuffer = &pNNFF->inputs.data;
	}
	else
	if (OffsetType_Output == nOffsetType)
	{
		pBuffer = &pNNFF->outputs.data;
	}
	else
	{
		return NULL;
	}

	if (pBuffer->cbSize <= cbOffset)
	{
		return NULL;
	}
	if (pBuffer->cbSize < cbOffset + cbSize)
	{
		return NULL;
	}

	return (n_uint8 *)pBuffer->pBuffer + cbOffset;
}

void OpenCLComputingEngine::ApplyActivation(n_activation_function nActivationFunction, n_uint8 cArgs, n_value *args, 
                                             n_neurons_count cRows, n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	if (N_ACTIVATION_IDENTITY == nActivationFunction)
	{
		ApplyActivationIdentity(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SIGMOID == nActivationFunction)
	{
		ApplyActivationSigmoid(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_BINARYSTEP == nActivationFunction)
	{
		ApplyActivationBinaryStep(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_TANH == nActivationFunction)
	{
		ApplyActivationTanH(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ARCTAN == nActivationFunction)
	{
		ApplyActivationArcTan(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTSIGN == nActivationFunction)
	{
		ApplyActivationSoftSign(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ISRU == nActivationFunction)
	{
		ApplyActivationISRU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_RELU == nActivationFunction)
	{
		ApplyActivationRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_LEAKYRELU == nActivationFunction)
	{
		ApplyActivationLeakyRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_PRELU == nActivationFunction)
	{
		ApplyActivationPRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ELU == nActivationFunction)
	{
		ApplyActivationELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SELU == nActivationFunction)
	{
		ApplyActivationSELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SRELU == nActivationFunction)
	{
		ApplyActivationSRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ISRLU == nActivationFunction)
	{
		ApplyActivationISRLU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTPLUS == nActivationFunction)
	{
		ApplyActivationSoftPlus(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_BENTIDENTITY == nActivationFunction)
	{
		ApplyActivationBentIdentity(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTEXPONENTIAL == nActivationFunction)
	{
		ApplyActivationSoftExponential(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SINUSOID == nActivationFunction)
	{
		ApplyActivationSinusoid(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SINC == nActivationFunction)
	{
		ApplyActivationSINC(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_GAUSSIAN == nActivationFunction)
	{
		ApplyActivationGaussian(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTMAX == nActivationFunction)
	{
		ApplyActivationSoftMax(cArgs, args, cRows, cCols, pOp1, pRes);
	}
}

void OpenCLComputingEngine::ApplyActivationDerivative(n_activation_function nActivationFunction, n_uint8 cArgs, 
                                                       n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
                                                       n_value *pOp1, n_value *pRes)
{
	if (N_ACTIVATION_IDENTITY == nActivationFunction)
	{
		ApplyActivationDerivativeIdentity(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SIGMOID == nActivationFunction)
	{
		ApplyActivationDerivativeSigmoid(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_BINARYSTEP == nActivationFunction)
	{
		ApplyActivationDerivativeBinaryStep(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_TANH == nActivationFunction)
	{
		ApplyActivationDerivativeTanH(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ARCTAN == nActivationFunction)
	{
		ApplyActivationDerivativeArcTan(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTSIGN == nActivationFunction)
	{
		ApplyActivationDerivativeSoftSign(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ISRU == nActivationFunction)
	{
		ApplyActivationDerivativeISRU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_RELU == nActivationFunction)
	{
		ApplyActivationDerivativeRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_LEAKYRELU == nActivationFunction)
	{
		ApplyActivationDerivativeLeakyRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_PRELU == nActivationFunction)
	{
		ApplyActivationDerivativePRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ELU == nActivationFunction)
	{
		ApplyActivationDerivativeELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SELU == nActivationFunction)
	{
		ApplyActivationDerivativeSELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SRELU == nActivationFunction)
	{
		ApplyActivationDerivativeSRELU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_ISRLU == nActivationFunction)
	{
		ApplyActivationDerivativeISRLU(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTPLUS == nActivationFunction)
	{
		ApplyActivationDerivativeSoftPlus(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_BENTIDENTITY == nActivationFunction)
	{
		ApplyActivationDerivativeBentIdentity(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTEXPONENTIAL == nActivationFunction)
	{
		ApplyActivationDerivativeSoftExponential(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SINUSOID == nActivationFunction)
	{
		ApplyActivationDerivativeSinusoid(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SINC == nActivationFunction)
	{
		ApplyActivationDerivativeSINC(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_GAUSSIAN == nActivationFunction)
	{
		ApplyActivationDerivativeGaussian(cArgs, args, cRows, cCols, pOp1, pRes);
	}
	else
	if (N_ACTIVATION_SOFTMAX == nActivationFunction)
	{
		ApplyActivationDerivativeSoftMax(cArgs, args, cRows, cCols, pOp1, pRes);
	}
}

void OpenCLComputingEngine::ApplyError(n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
                                        n_neurons_count cRows, n_neurons_count cCols, n_value *pOp1, n_value *pOp2, 
                                        n_value *pnError)
{
	if (N_ERROR_SQUARE == nErrorFunction)
	{
		ApplyErrorSquare(cArgs, args, cRows, cCols, pOp1, pOp2, pnError);
	}
	else
	if (N_ERROR_CROSSENTROPY == nErrorFunction)
	{
		ApplyErrorCrossEntropy(cArgs, args, cRows, cCols, pOp1, pOp2, pnError);
	}
}

void OpenCLComputingEngine::ApplyErrorDerivative(n_error_function nErrorFunction, n_uint8 cArgs, n_value *args,
                                                  n_neurons_count cRows, n_neurons_count cCols, n_value *pOp1, 
                                                  n_value *pOp2, n_value *pRes)
{
	if (N_ERROR_SQUARE == nErrorFunction)
	{
		ApplyErrorDerivativeSquare(cArgs, args, cRows, cCols, pOp1, pOp2, pRes);
	}
	else
	if (N_ERROR_CROSSENTROPY == nErrorFunction)
	{
		ApplyErrorDerivativeCrossEntropy(cArgs, args, cRows, cCols, pOp1, pOp2, pRes);
	}
}

const Char *OpenCLComputingEngine::GetName()
{
	return NAME();
}

Status OpenCLComputingEngine::InitLayersAndLinks(n_fnn *pNNFF)
{
	pNNFF->layers.pCECTX = NULL;
	pNNFF->links.pCECTX  = NULL;

	return Status();
}

Status OpenCLComputingEngine::UninitLayersAndLinks(n_fnn *pNNFF)
{
	pNNFF->layers.pCECTX = NULL;
	pNNFF->links.pCECTX  = NULL;

	return Status();
}

Status OpenCLComputingEngine::InitInputsAndOutputs(n_fnn *pNNFF)
{
	pNNFF->inputs.pCECTX  = NULL;
	pNNFF->outputs.pCECTX = NULL;

	return Status();
}

Status OpenCLComputingEngine::UninitInputsAndOutputs(n_fnn *pNNFF)
{
	pNNFF->inputs.pCECTX  = NULL;
	pNNFF->outputs.pCECTX = NULL;

	return Status();
}

Status OpenCLComputingEngine::Assign(n_fnn *pNNFF,
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	for (n_neurons_count cRow = 0; cRow < cOp1Rows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cOp1Cols; cCol++)
		{
			*pRes = *pOp1;
			pOp1++;
			pRes++;
		}
	}

	return Status();
}

Status OpenCLComputingEngine::Add(n_fnn *pNNFF,
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pOp2;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cOp2Rows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cOp2Cols)
	{
		return Status_InvalidArg;
	}
	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pOp2 = (n_value *)GetPtr(pNNFF, nOp2OffsetType, cbOp2Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	for (n_neurons_count cRow = 0; cRow < cOp1Rows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cOp1Cols; cCol++)
		{
			*pRes = *pOp1 + *pOp2;
			pOp1++;
			pOp2++;
			pRes++;
		}
	}

	return Status();
}

Status OpenCLComputingEngine::Substract(n_fnn *pNNFF,
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pOp2;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cOp2Rows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cOp2Cols)
	{
		return Status_InvalidArg;
	}
	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pOp2 = (n_value *)GetPtr(pNNFF, nOp2OffsetType, cbOp2Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	for (n_neurons_count cRow = 0; cRow < cOp1Rows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cOp1Cols; cCol++)
		{
			*pRes = *pOp1 - *pOp2;
			pOp1++;
			pOp2++;
			pRes++;
		}
	}

	return Status();
}

Status OpenCLComputingEngine::Multiply(n_fnn *pNNFF,
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pOp2;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cOp2Rows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cOp2Cols)
	{
		return Status_InvalidArg;
	}
	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pOp2 = (n_value *)GetPtr(pNNFF, nOp2OffsetType, cbOp2Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	for (n_neurons_count cRow = 0; cRow < cOp1Rows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cOp1Cols; cCol++)
		{
			*pRes = *pOp1 * *pOp2;
			pOp1++;
			pOp2++;
			pRes++;
		}
	}

	return Status();
}

//cOp1Rows = cOp2Rows = 1
//res[i,j] = op2[i] * op1[j]
Status OpenCLComputingEngine::OuterProduct(n_fnn *pNNFF,
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pOp2;
	n_value   *pRes;
	n_size    cbOp1Size;
	n_size    cbOp2Size;
	n_size    cbResSize;

	if (1 != cOp1Rows)
	{
		return Status_InvalidArg;
	}
	if (1 != cOp2Cols)
	{
		return Status_InvalidArg;
	}
	if (cOp2Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbOp1Size = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbOp1Size)))
	{
		return Status_InvalidArg;
	}
	cbOp2Size = sizeof(n_value) * cOp2Rows * cOp2Cols;
	if (NULL == (pOp2 = (n_value *)GetPtr(pNNFF, nOp2OffsetType, cbOp2Offset, cbOp2Size)))
	{
		return Status_InvalidArg;
	}
	cbResSize = sizeof(n_value) * cResRows * cResCols;
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbResSize)))
	{
		return Status_InvalidArg;
	}

	for (n_neurons_count cRow = 0; cRow < cOp2Rows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cOp1Cols; cCol++)
		{
			*pRes = pOp2[cRow] * pOp1[cCol];
			pRes++;
		}
	}

	return Status();
}

Status OpenCLComputingEngine::Multiply(n_fnn *pNNFF,
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            n_value fOp2,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	for (n_neurons_count cRow = 0; cRow < cOp1Rows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cOp1Cols; cCol++)
		{
			*pRes = *pOp1 * fOp2;
			pOp1++;
			pRes++;
		}
	}

	return Status();
}

Status OpenCLComputingEngine::DotProduct(n_fnn *pNNFF,
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pOp2;
	n_value   *pRes;
	n_value   fTmp;
	n_size    cbOp1Size;
	n_size    cbOp2Size;
	n_size    cbResSize;

	if (cOp1Cols != cOp2Rows)
	{
		return Status_InvalidArg;
	}
	if (cOp2Cols != cResCols)
	{
		return Status_InvalidArg;
	}
	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}

	cbOp1Size = sizeof(n_value) * cOp1Rows * cOp1Cols;
	cbOp2Size = sizeof(n_value) * cOp2Rows * cOp2Cols;
	cbResSize = sizeof(n_value) * cResRows * cResCols;

	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbOp1Size)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pOp2 = (n_value *)GetPtr(pNNFF, nOp2OffsetType, cbOp2Offset, cbOp2Size)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbResSize)))
	{
		return Status_InvalidArg;
	}

	for (n_neurons_count cRow = 0; cRow < cOp1Rows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cOp2Cols; cCol++)
		{
			fTmp = N_VAL(0.0);

			for (n_neurons_count i = 0; i < cOp1Cols; i++)
			{
				fTmp += pOp1[cRow * cOp1Cols + i] * pOp2[i * cOp2Cols + cCol];
			}

			pRes[cRow * cOp2Cols + cCol] = fTmp;
		}
	}

	return Status();
}


Status OpenCLComputingEngine::ApplyActivation(n_fnn *pNNFF, 
            n_activation_function nActivateFunction, n_uint8 cArgs, n_value *args, 
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	ApplyActivation(nActivateFunction, cArgs, args, cOp1Rows, cOp1Cols, pOp1, pRes);

	return Status();
}

Status OpenCLComputingEngine::ApplyActivationDerivative(n_fnn *pNNFF, 
            n_activation_function nActivateFunction, n_uint8 cArgs, n_value *args, 
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cResRows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cResCols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	ApplyActivationDerivative(nActivateFunction, cArgs, args, cOp1Rows, cOp1Cols, pOp1, pRes);

	return Status();
}

Status OpenCLComputingEngine::ApplyError(n_fnn *pNNFF, 
            n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols, 
            n_value *pnError)
{
	n_value   *pOp1;
	n_value   *pOp2;
	n_size    cbSize;

	if (cOp1Rows != cOp2Rows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cOp2Cols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pOp2 = (n_value *)GetPtr(pNNFF, nOp2OffsetType, cbOp2Offset, cbSize)))
	{
		return Status_InvalidArg;
	}

	ApplyError(nErrorFunction, cArgs, args, cOp1Rows, cOp1Cols, pOp1, pOp2, pnError);

	return Status();
}

Status OpenCLComputingEngine::ApplyErrorDerivative(n_fnn *pNNFF, 
            n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols, 
            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols)
{
	n_value   *pOp1;
	n_value   *pOp2;
	n_value   *pRes;
	n_size    cbSize;

	if (cOp1Rows != cOp2Rows)
	{
		return Status_InvalidArg;
	}
	if (cOp1Cols != cOp2Cols)
	{
		return Status_InvalidArg;
	}

	cbSize = sizeof(n_value) * cOp1Rows * cOp1Cols;
	if (NULL == (pOp1 = (n_value *)GetPtr(pNNFF, nOp1OffsetType, cbOp1Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pOp2 = (n_value *)GetPtr(pNNFF, nOp2OffsetType, cbOp2Offset, cbSize)))
	{
		return Status_InvalidArg;
	}
	if (NULL == (pRes = (n_value *)GetPtr(pNNFF, nResOffsetType, cbResOffset, cbSize)))
	{
		return Status_InvalidArg;
	}

	ApplyErrorDerivative(nErrorFunction, cArgs, args, cOp1Rows, cOp1Cols, pOp1, pOp2, pRes);

	return Status();
}

void OpenCLComputingEngine::ApplyActivationIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                     n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = *pOp1;
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                               n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	pOp1;

	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = N_VAL(1.0);
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSigmoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                    n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = N_VAL(1.0) / (N_VAL(1.0) + (n_value)exp(-*pOp1));
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSigmoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                              n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = *pOp1 * (N_VAL(1.0) - *pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationBinaryStep(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                       n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = N_VAL(0.0);
			}
			else
			{
				*pRes = N_VAL(1.0);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeBinaryStep(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                                 n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = N_VAL(0.0);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationTanH(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                 n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = tanh(*pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeTanH(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = N_VAL(1.0) - tanh(*pOp1) * tanh(*pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationArcTan(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                   n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = atan(*pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeArcTan(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                             n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = N_VAL(1.0) / (*pOp1 * *pOp1 + N_VAL(1.0));
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSoftSign(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                     n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = *pOp1 / (N_VAL(1.0) + abs(*pOp1));
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSoftSign(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                               n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = N_VAL(1.0) / ((N_VAL(1.0) + abs(*pOp1)) * (N_VAL(1.0) + abs(*pOp1)));
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationISRU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                 n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(1.0);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = *pOp1 / sqrt(N_VAL(1.0) + fAlpha * *pOp1 * *pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeISRU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fTmp;
	n_value fAlpha = N_VAL(1.0);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			fTmp = N_VAL(1.0) / sqrt(N_VAL(1.0) + fAlpha * *pOp1 * *pOp1);
			*pRes = fTmp * fTmp * fTmp;
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                 n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = N_VAL(0.0);
			}
			else
			{
				*pRes = *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = N_VAL(0.0);
			}
			else
			{
				*pRes = N_VAL(1.0);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationLeakyRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                      n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = N_VAL(0.01) * *pOp1;
			}
			else
			{
				*pRes = *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeLeakyRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
	                                                             n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = N_VAL(0.01);
			}
			else
			{
				*pRes = N_VAL(1.0);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationPRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                  n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.5);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = fAlpha * *pOp1;
			}
			else
			{
				*pRes = *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativePRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                            n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.5);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = fAlpha;
			}
			else
			{
				*pRes = N_VAL(1.0);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.01);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = fAlpha * (exp(*pOp1) - N_VAL(1.0));
			}
			else
			{
				*pRes = *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                          n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.01);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = fAlpha * (exp(*pOp1) - N_VAL(1.0)) + fAlpha;
			}
			else
			{
				*pRes = N_VAL(1.0);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                 n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha  = N_VAL(1.67326);
	n_value fLambda = N_VAL(1.0507);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	if (2 <= cArgs)
	{
		fLambda = args[1];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = fLambda * fAlpha * (exp(*pOp1) - 1);
			}
			else
			{
				*pRes = fLambda * *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(1.67326);
	n_value fLambda = N_VAL(1.0507);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	if (2 <= cArgs)
	{
		fLambda = args[1];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = fLambda * fAlpha * exp(*pOp1);
			}
			else
			{
				*pRes = fLambda;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                  n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fTl = N_VAL(-0.4);
	n_value fAl = N_VAL(0.4);
	n_value fTr = N_VAL(0.4);
	n_value fAr = N_VAL(2.0);

	if (1 <= cArgs)
	{
		fTl = args[0];
	}
	if (2 <= cArgs)
	{
		fAl = args[1];
	}
	if (3 <= cArgs)
	{
		fTr = args[2];
	}
	if (4 <= cArgs)
	{
		fAr = args[3];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (*pOp1 <= fTl)
			{
				*pRes = fTl + fAl * (*pOp1 - fTl);
			}
			else
			if (*pOp1 >= fTr)
			{
				*pRes = fTr + fAr * (*pOp1 - fTr);
			}
			else
			{
				*pRes = *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                            n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fTl = N_VAL(-0.4);
	n_value fAl = N_VAL(0.4);
	n_value fTr = N_VAL(0.4);
	n_value fAr = N_VAL(2.0);

	if (1 <= cArgs)
	{
		fTl = args[0];
	}
	if (2 <= cArgs)
	{
		fAl = args[1];
	}
	if (3 <= cArgs)
	{
		fTr = args[2];
	}
	if (4 <= cArgs)
	{
		fAr = args[3];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (*pOp1 <= fTl)
			{
				*pRes = fAl;
			}
			else
			if (*pOp1 >= fTr)
			{
				*pRes = fAr;
			}
			else
			{
				*pRes = N_VAL(1.0);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationISRLU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                  n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.01);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				*pRes = *pOp1 / sqrt(1 + fAlpha * *pOp1 * *pOp1);
			}
			else
			{
				*pRes = *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeISRLU(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                            n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.01);
	n_value fTmp;

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > *pOp1)
			{
				fTmp = N_VAL(1.0) / sqrt(N_VAL(1.0) + fAlpha * *pOp1 * *pOp1);
				*pRes = fTmp * fTmp + fTmp;
			}
			else
			{
				*pRes = N_VAL(1.0);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSoftPlus(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                     n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = log(N_VAL(1.0) + *pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSoftPlus(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                               n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = N_VAL(1.0) / (N_VAL(1.0) + exp(- *pOp1));
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationBentIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                         n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.01);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = (sqrt(*pOp1 * *pOp1 + 1) - N_VAL(1.0)) / N_VAL(2.0) + *pOp1;
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeBentIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                                   n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = *pOp1 / (N_VAL(2.0) * sqrt(*pOp1 * *pOp1 + N_VAL(1.0))) + N_VAL(1.0);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSoftExponential(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                            n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.01);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > fAlpha)
			{
				*pRes = -log(N_VAL(1.0) - fAlpha * (*pOp1 + fAlpha)) / fAlpha;
			}
			else
			if (N_VAL(0.0) < fAlpha)
			{
				*pRes = (exp(fAlpha * *pOp1) - N_VAL(1.0)) / fAlpha + fAlpha;
			}
			else
			{
				*pRes = *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSoftExponential(n_uint8 cArgs, n_value *args, 
                                                                      n_neurons_count cRows, n_neurons_count cCols, 
                                                                      n_value *pOp1, n_value *pRes)
{
	n_value fAlpha = N_VAL(0.01);

	if (1 <= cArgs)
	{
		fAlpha = args[0];
	}
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) > fAlpha)
			{
				*pRes = N_VAL(1.0) / (N_VAL(1.0) - fAlpha * (fAlpha + *pOp1));
			}
			else
			{
				*pRes = exp(fAlpha * *pOp1);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSinusoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                     n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = sin(*pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSinusoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                               n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = cos(*pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSINC(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                 n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) == *pOp1)
			{
				*pRes = N_VAL(1.0);
			}
			else
			{
				*pRes = sin(*pOp1) / *pOp1;
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSINC(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			if (N_VAL(0.0) == *pOp1)
			{
				*pRes = N_VAL(0.0);
			}
			else
			{
				*pRes = cos(*pOp1) / *pOp1 - sin(*pOp1) / (*pOp1 * *pOp1);
			}
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationGaussian(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                     n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = exp(- *pOp1 * *pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeGaussian(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                               n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = - N_VAL(2.0) * *pOp1 * exp(- *pOp1 * *pOp1);
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationSoftMax(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                    n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value *pOp1Tmp = pOp1;
	n_value *pResTmp = pRes;
	n_value fSum;

	fSum = N_VAL(0.0);
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			fSum += exp(*pOp1Tmp);
			pOp1Tmp++;
			pResTmp++;
		}
	}

	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = exp(*pOp1Tmp) / fSum;
			pOp1++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyActivationDerivativeSoftMax(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                              n_neurons_count cCols, n_value *pOp1, n_value *pRes)
{
	n_value *pOp1Tmp = pOp1;
	n_value *pResTmp = pRes;
	n_value fSum;

	fSum = N_VAL(0.0);
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			fSum += exp(*pOp1Tmp);
			pOp1Tmp++;
			pResTmp++;
		}
	}

	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes = exp(*pOp1Tmp) * (fSum - exp(*pOp1Tmp)) / (fSum * fSum);
			pOp1++;
			pRes++;
		}
	}
}

//===

void OpenCLComputingEngine::ApplyErrorSquare(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                              n_neurons_count cCols, n_value *pOp1, n_value *pOp2, n_value *pnError)
{
	*pnError = N_VAL(0.0);
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pnError += N_VAL(0.5) * (*pOp2 - *pOp1) * (*pOp2 - *pOp1);
			pOp1++;
			pOp2++;
		}
	}
}

//===

void OpenCLComputingEngine::ApplyErrorDerivativeSquare(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                        n_neurons_count cCols, n_value *pOp1, n_value *pOp2, 
                                                        n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes += -(*pOp2 - *pOp1);
			pOp1++;
			pOp2++;
			pRes++;
		}
	}
}

void OpenCLComputingEngine::ApplyErrorCrossEntropy(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
                                                    n_neurons_count cCols, n_value *pOp1, n_value *pOp2, 
                                                    n_value *pnError)
{
	*pnError = N_VAL(0.0);
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pnError += *pOp2 * log10(*pOp1) + (N_VAL(1.0) - *pOp2) * log10(N_VAL(1.0) - *pOp1);
			pOp1++;
			pOp2++;
		}
	}
	*pnError = -*pnError;
}

void OpenCLComputingEngine::ApplyErrorDerivativeCrossEntropy(n_uint8 cArgs, n_value *args, n_neurons_count cRows,
                                                              n_neurons_count cCols, n_value *pOp1, n_value *pOp2, 
                                                              n_value *pRes)
{
	for (n_neurons_count cRow = 0; cRow < cRows; cRow++)
	{
		for (n_neurons_count cCol = 0; cCol < cCols; cCol++)
		{
			*pRes += - N_VAL(1.0) * ((*pOp2 * (N_VAL(1.0) / *pOp1)) + 
			                         (N_VAL(1.0) - *pOp2) * (N_VAL(1.0) / (N_VAL(1.0) - *pOp1)));
			pOp1++;
			pOp2++;
			pRes++;
		}
	}
}

}//namespace FNN

}//namespace NeuTron
