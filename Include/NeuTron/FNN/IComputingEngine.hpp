
#pragma once


#include "NeuTron/Platform/FNN.h"
#include "CX/Types.hpp"
#include "CX/Status.hpp"


namespace NeuTron
{

namespace FNN
{

class IComputingEngine
{
public:

	enum OffsetType
	{
		OffsetType_Layer,
		OffsetType_Link,
		OffsetType_Input,
		OffsetType_Output,
	};

	virtual ~IComputingEngine() { }

	virtual const CX::Char *GetName() = 0;

	virtual CX::Status InitLayersAndLinks(n_fnn *pNNFF) = 0;

	virtual CX::Status UninitLayersAndLinks(n_fnn *pNNFF) = 0;

	virtual CX::Status InitInputsAndOutputs(n_fnn *pNNFF) = 0;

	virtual CX::Status UninitInputsAndOutputs(n_fnn *pNNFF) = 0;

	virtual CX::Status Assign(n_fnn *pNNFF,
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status Add(n_fnn *pNNFF,
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status Substract(n_fnn *pNNFF,
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status Multiply(n_fnn *pNNFF,
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	//cOp1Rows = cOp2Rows = 1
	//res[i,j] = op2[i] * op1[j]
	virtual CX::Status OuterProduct(n_fnn *pNNFF,
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status Multiply(n_fnn *pNNFF,
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            n_value fOp2,
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status DotProduct(n_fnn *pNNFF,
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols,
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status ApplyActivation(n_fnn *pNNFF, 
	            n_activation_function nActivateFunction, n_uint8 cArgs, n_value *args, 
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status ApplyActivationDerivative(n_fnn *pNNFF, 
	            n_activation_function nActivateFunction, n_uint8 cArgs, n_value *args, 
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

	virtual CX::Status ApplyError(n_fnn *pNNFF, 
	            n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols, 
	            n_value *pnError) = 0;

	virtual CX::Status ApplyErrorDerivative(n_fnn *pNNFF, 
	            n_error_function nErrorFunction, n_uint8 cArgs, n_value *args, 
	            OffsetType nOp1OffsetType, n_offset cbOp1Offset, n_neurons_count cOp1Rows, n_neurons_count cOp1Cols, 
	            OffsetType nOp2OffsetType, n_offset cbOp2Offset, n_neurons_count cOp2Rows, n_neurons_count cOp2Cols, 
	            OffsetType nResOffsetType, n_offset cbResOffset, n_neurons_count cResRows, n_neurons_count cResCols) = 0;

};

}//namespace FNN

}//namespace NeuTron

