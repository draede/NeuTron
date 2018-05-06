
#pragma once


#include "CX/Types.hpp"
#include "CX/Status.hpp"
#include "NeuTron/DataSet.hpp"


namespace NeuTron
{

namespace MNIST
{

class Converter
{
public:

	static CX::Status ConvertImages(const CX::Char *szInputPath, const CX::Char *szOutputPath, 
	                                n_value fMin = N_VAL(0.01), n_value fMax = N_VAL(0.99));

	static CX::Status ConvertLabels(const CX::Char *szInputPath, const CX::Char *szOutputPath, 
	                                n_value fNotMatched = N_VAL(0.0), n_value fMatched = N_VAL(1.0));

private:

	static const CX::UInt32   IMAGES_MAGIC          = 0x00000803;
	static const CX::UInt32   LABELS_MAGIC          = 0x00000801;

	static const CX::UInt32   IMAGES_ROWS           = 28;
	static const CX::UInt32   IMAGES_COLUMNS        = 28;
	static const CX::UInt32   IMAGES_COUNT          = IMAGES_ROWS * IMAGES_COLUMNS;
	static const CX::UInt32   LABELS_COUNT          = 10;

	static const CX::Size     IMAGES_BUFFER_ITEMS    = 100;
	static const CX::Size     IMAGES_BUFFER_IN_SIZE  = IMAGES_BUFFER_ITEMS * IMAGES_COUNT * sizeof(CX::Byte);
	static const CX::Size     IMAGES_BUFFER_OUT_SIZE = IMAGES_BUFFER_ITEMS * IMAGES_COUNT * sizeof(n_value);

	static const CX::Size     LABELS_BUFFER_ITEMS    = 16384;
	static const CX::Size     LABELS_BUFFER_IN_SIZE  = LABELS_BUFFER_ITEMS * sizeof(CX::Byte);
	static const CX::Size     LABELS_BUFFER_OUT_SIZE = LABELS_BUFFER_ITEMS * LABELS_COUNT * sizeof(n_value);

	Converter();

	~Converter();

};

}//namespace MNIST

}//namespace NeuTron
