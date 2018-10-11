
#pragma once


#include "NeuTron/FNN/IComputingEngine.hpp"
#include "CX/Types.hpp"
#include "CX/Status.hpp"


namespace NeuTron
{

namespace FNN
{

class OpenCLComputingEngine : public IComputingEngine
{
public:

	static const CX::Char *NAME()   { return "NeuTron.FNN.OpenCLComputingEngine"; }

	OpenCLComputingEngine();

	~OpenCLComputingEngine();

	virtual const CX::Char *GetName();

	virtual CX::Status InitLayersAndLinks(n_fnn *pNNFF);

	virtual CX::Status UninitLayersAndLinks(n_fnn *pNNFF);

	virtual CX::Status InitInputsAndOutputs(n_fnn *pNNFF);

	virtual CX::Status UninitInputsAndOutputs(n_fnn *pNNFF);

	virtual CX::Status Assign(n_fnn *pNNFF,
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status Add(n_fnn *pNNFF,
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status Substract(n_fnn *pNNFF,
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status Multiply(n_fnn *pNNFF,
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	//cOp1Rows = cOp2Rows = 1
	//res[i,j] = op2[i,0] * op1[0,j]
	virtual CX::Status OuterProduct(n_fnn *pNNFF,
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status Multiply(n_fnn *pNNFF,
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              n_value fOp2,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status DotProduct(n_fnn *pNNFF,
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status ApplyActivation(n_fnn *pNNFF, 
	              n_activation_function nActivateFunction, n_uint8 cArgs, n_value *args, 
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols,
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status ApplyActivationDerivative(n_fnn *pNNFF, 
	              n_activation_function nActivateFunction, n_uint8 cArgs, n_value *args, 
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

	virtual CX::Status ApplyError(n_fnn *pNNFF, 
	              n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	              OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols, 
	              n_value *pnError);

	virtual CX::Status ApplyErrorDerivative(n_fnn *pNNFF, 
	              n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
	              OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	              OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols, 
	              OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols);

private:

	void *GetPtr(n_fnn *pNNFF, OffsetType nOffsetType, n_offset cbOffset, n_size cbSize);

	static void ApplyActivation(n_activation_function nActivationFunction, n_uint8 cArgs, n_value *args, 
	                            n_neurons_count cRows, n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivative(n_activation_function nActivationFunction, n_uint8 cArgs, n_value *args, 
	                                      n_neurons_count cRows, n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyError(n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                       n_neurons_count cCols, n_value *pOp1, n_value *pOp2, n_value *pnError);

	static void ApplyErrorDerivative(n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
	                                 n_neurons_count cRows, n_neurons_count cCols, n_value *pOp1, n_value *pOp2, 
	                                 n_value *pnError);

	//===

	static void ApplyActivationIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                    n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                              n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSigmoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                   n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSigmoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                             n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationBinaryStep(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                      n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeBinaryStep(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                                n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationTanH(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeTanH(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                          n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationArcTan(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                  n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeArcTan(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                            n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSoftSign(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                   n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSoftSign(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                              n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationISRU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeISRU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                          n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                          n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationLeakyRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                     n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeLeakyRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                               n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationPRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                 n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativePRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                               n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                         n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                          n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                 n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSRELU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationISRLU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                 n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeISRLU(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSoftPlus(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                    n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSoftPlus(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                              n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationBentIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                        n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeBentIdentity(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                                  n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSoftExponential(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                           n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSoftExponential(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                                     n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSinusoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                    n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSinusoid(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                               n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSINC(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSINC(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                          n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationGaussian(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                    n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeGaussian(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                              n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	static void ApplyActivationSoftMax(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                   n_value *pOp1, n_value *pRes);

	static void ApplyActivationDerivativeSoftMax(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                             n_neurons_count cCols, n_value *pOp1, n_value *pRes);

	//===

	static void ApplyErrorSquare(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                             n_value *pOp1, n_value *pOp2, n_value *pnError);

	static void ApplyErrorDerivativeSquare(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                                       n_value *pOp1, n_value *pOp2, n_value *pRes);

	static void ApplyErrorCrossEntropy(n_uint8 cArgs, n_value *args, n_neurons_count cRows, n_neurons_count cCols, 
	                             n_value *pOp1, n_value *pOp2, n_value *pnError);

	static void ApplyErrorDerivativeCrossEntropy(n_uint8 cArgs, n_value *args, n_neurons_count cRows, 
	                                             n_neurons_count cCols, n_value *pOp1, n_value *pOp2, n_value *pRes);

};

}//namespace FNN

}//namespace NeuTron

