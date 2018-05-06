
#include "NeuTron/MNIST/Converter.hpp"
#include "CX/Util/MemPool.hpp"
#include "CX/IO/FileInputStream.hpp"
#include "CX/IO/FileOutputStream.hpp"
#include "CX/Sys/ByteOrder.hpp"


using namespace CX;


namespace NeuTron
{

namespace MNIST
{

Converter::Converter()
{
}

Converter::~Converter()
{
}

/*
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
*/
Status Converter::ConvertImages(const Char *szInputPath, const Char *szOutputPath, n_value fMin/* = N_VAL(0.01)*/, 
                                n_value fMax/* = N_VAL(0.99)*/)
{
	Util::DynMemPool       memin;
	Util::DynMemPool       memout;
	IO::FileInputStream    is(szInputPath);
	IO::FileOutputStream   os(szOutputPath);
	Size                   cbSize;
	Size                   cbSizeTmp;
	const Byte             *pMemIn;
	n_value                *pMemOut;
	UInt32                 nMagic;
	UInt32                 cNoOfImages;
	UInt32                 cNoOfImagesTmp;
	UInt32                 cNoOfRows;
	UInt32                 cNoOfColumns;
	UInt32                 cTmp;
	Status                 status;

	if (!(status = memin.SetSize(IMAGES_BUFFER_IN_SIZE)))
	{
		return Status(Status_MemAllocFailed, "Failed to allocate {1} bytes", IMAGES_BUFFER_IN_SIZE);
	}
	if (!(status = memout.SetSize(IMAGES_BUFFER_OUT_SIZE)))
	{
		return Status(Status_MemAllocFailed, "Failed to allocate {1} bytes", IMAGES_BUFFER_OUT_SIZE);
	}
	if (!is.IsOK())
	{
		return Status(Status_OpenFailed, "Failed to open '{1}'", szInputPath);
	}
	if (!os.IsOK())
	{
		return Status(Status_CreateFailed, "Failed to create '{1}'", szOutputPath);
	}
	for (;;)
	{
		if (!(status = is.Read(memin.GetMem(), sizeof(UInt32) * 4, &cbSize)))
		{
			break;
		}
		if (sizeof(UInt32) * 4 != cbSize)
		{
			status = Status(Status_ParseFailed, "Failed to read header");

			break;
		}
		pMemIn = (const Byte *)memin.GetMem();
		memcpy(&nMagic, pMemIn, sizeof(UInt32)); pMemIn += sizeof(UInt32);
		nMagic = Sys::ByteOrder::BE2H(nMagic);
		if (IMAGES_MAGIC != nMagic)
		{
			status = Status(Status_ParseFailed, "Invalid magic number");

			break;
		}
		memcpy(&cNoOfImages, pMemIn, sizeof(UInt32)); pMemIn += sizeof(UInt32);
		cNoOfImages = Sys::ByteOrder::BE2H(cNoOfImages);
		if (0 == cNoOfImages)
		{
			status = Status(Status_ParseFailed, "Invalid number of images");

			break;
		}
		memcpy(&cNoOfRows, pMemIn, sizeof(UInt32)); pMemIn += sizeof(UInt32);
		cNoOfRows = Sys::ByteOrder::BE2H(cNoOfRows);
		if (IMAGES_ROWS != cNoOfRows)
		{
			status = Status(Status_ParseFailed, "Invalid number of rows");

			break;
		}
		memcpy(&cNoOfColumns, pMemIn, sizeof(UInt32)); pMemIn += sizeof(UInt32);
		cNoOfColumns = Sys::ByteOrder::BE2H(cNoOfColumns);
		if (IMAGES_COLUMNS != cNoOfColumns)
		{
			status = Status(Status_ParseFailed, "Invalid number of columns");

			break;
		}

		if (sizeof(n_float) == sizeof(n_value))
		{
			if (!(status = os.Write(&DataSet::FLOAT_FILE_MAGIC, sizeof(DataSet::FLOAT_FILE_MAGIC), &cbSizeTmp)))
			{
				break;
			}
		}
		else
		{
			if (!(status = os.Write(&DataSet::DOUBLE_FILE_MAGIC, sizeof(DataSet::DOUBLE_FILE_MAGIC), &cbSizeTmp)))
			{
				break;
			}
		}
		cTmp = Sys::ByteOrder::H2LE(cNoOfImages);
		if (!(status = os.Write(&cTmp, sizeof(cTmp), &cbSizeTmp)))
		{
			break;
		}
		cTmp = Sys::ByteOrder::H2LE(IMAGES_COUNT);
		if (!(status = os.Write(&cTmp, sizeof(cTmp), &cbSizeTmp)))
		{
			break;
		}

		while (0 < cNoOfImages)
		{
			if (cNoOfImages >= IMAGES_BUFFER_ITEMS)
			{
				cNoOfImagesTmp = IMAGES_BUFFER_ITEMS;
			}
			else
			{
				cNoOfImagesTmp = cNoOfImages;
			}
			if (!(status = is.Read(memin.GetMem(), cNoOfImagesTmp * IMAGES_COUNT * sizeof(Byte), &cbSize)))
			{
				break;
			}
			if (cNoOfImagesTmp * IMAGES_COUNT * sizeof(Byte) != cbSize)
			{
				status = Status(Status_ParseFailed, "Failed to read data");

				break;
			}

			pMemIn  = (const Byte *)memin.GetMem();
			pMemOut = (n_value *)memout.GetMem();
			for (UInt32 i = 0; i < cNoOfImagesTmp; i++)
			{
				for (UInt32 cRow = 0; cRow < IMAGES_ROWS; cRow++)
				{
					for (UInt32 cColumn = 0; cColumn < IMAGES_COLUMNS; cColumn++)
					{
						*pMemOut = fMin + ((n_value)(*pMemIn) / N_VAL(255.0)) * (fMax - fMin);
						pMemOut++;
						pMemIn++;
					}
				}
			}
			if (!(status = os.Write(memout.GetMem(), cNoOfImagesTmp * IMAGES_COUNT * sizeof(n_value), &cbSizeTmp)))
			{
				break;
			}

			cNoOfImages -= cNoOfImagesTmp;
		}

		break;
	}

	return status;
}

/*
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
*/
Status Converter::ConvertLabels(const Char *szInputPath, const Char *szOutputPath, 
                                n_value fNotMatched/* = N_VAL(0.0)*/, n_value fMatched/* = N_VAL(1.0)*/)
{
	Util::DynMemPool       memin;
	Util::DynMemPool       memout;
	IO::FileInputStream    is(szInputPath);
	IO::FileOutputStream   os(szOutputPath);
	Size                   cbSize;
	Size                   cbSizeTmp;
	const Byte             *pMemIn;
	n_value                *pMemOut;
	UInt32                 nMagic;
	UInt32                 cNoOfLabels;
	UInt32                 cNoOfLabelsTmp;
	UInt32                 cTmp;
	Status                 status;

	if (!(status = memin.SetSize(LABELS_BUFFER_IN_SIZE)))
	{
		return Status(Status_MemAllocFailed, "Failed to allocate {1} bytes", LABELS_BUFFER_IN_SIZE);
	}
	if (!(status = memout.SetSize(LABELS_BUFFER_OUT_SIZE)))
	{
		return Status(Status_MemAllocFailed, "Failed to allocate {1} bytes", LABELS_BUFFER_OUT_SIZE);
	}
	if (!is.IsOK())
	{
		return Status(Status_OpenFailed, "Failed to open '{1}'", szInputPath);
	}
	if (!os.IsOK())
	{
		return Status(Status_CreateFailed, "Failed to create '{1}'", szOutputPath);
	}
	for (;;)
	{
		if (!(status = is.Read(memin.GetMem(), sizeof(UInt32) * 2, &cbSize)))
		{
			break;
		}
		if (sizeof(UInt32) * 2 != cbSize)
		{
			status = Status(Status_ParseFailed, "Failed to read header");

			break;
		}
		pMemIn = (const Byte *)memin.GetMem();
		memcpy(&nMagic, pMemIn, sizeof(UInt32)); pMemIn += sizeof(UInt32);
		nMagic = Sys::ByteOrder::BE2H(nMagic);
		if (LABELS_MAGIC != nMagic)
		{
			status = Status(Status_ParseFailed, "Invalid magic number");

			break;
		}
		memcpy(&cNoOfLabels, pMemIn, sizeof(UInt32)); pMemIn += sizeof(UInt32);
		cNoOfLabels = Sys::ByteOrder::BE2H(cNoOfLabels);
		if (0 == cNoOfLabels)
		{
			status = Status(Status_ParseFailed, "Invalid number of labels");

			break;
		}

		if (sizeof(n_float) == sizeof(n_value))
		{
			if (!(status = os.Write(&DataSet::FLOAT_FILE_MAGIC, sizeof(DataSet::FLOAT_FILE_MAGIC), &cbSizeTmp)))
			{
				break;
			}
		}
		else
		{
			if (!(status = os.Write(&DataSet::DOUBLE_FILE_MAGIC, sizeof(DataSet::DOUBLE_FILE_MAGIC), &cbSizeTmp)))
			{
				break;
			}
		}
		cTmp = Sys::ByteOrder::H2LE(cNoOfLabels);
		if (!(status = os.Write(&cTmp, sizeof(cTmp), &cbSizeTmp)))
		{
			break;
		}
		cTmp = Sys::ByteOrder::H2LE(LABELS_COUNT);
		if (!(status = os.Write(&cTmp, sizeof(cTmp), &cbSizeTmp)))
		{
			break;
		}

		while (0 < cNoOfLabels)
		{
			if (cNoOfLabels >= LABELS_BUFFER_ITEMS)
			{
				cNoOfLabelsTmp = LABELS_BUFFER_ITEMS;
			}
			else
			{
				cNoOfLabelsTmp = cNoOfLabels;
			}
			if (!(status = is.Read(memin.GetMem(), cNoOfLabelsTmp * sizeof(Byte), &cbSize)))
			{
				break;
			}
			if (cNoOfLabelsTmp * sizeof(Byte) != cbSize)
			{
				status = Status(Status_ParseFailed, "Failed to read data");

				break;
			}

			pMemIn  = (const Byte *)memin.GetMem();
			pMemOut = (n_value *)memout.GetMem();
			for (UInt32 i = 0; i < cNoOfLabelsTmp; i++)
			{
				for (Byte k = 0; k < 10; k++)
				{
					if (k == *pMemIn)
					{
						*pMemOut = fMatched;
					}
					else
					{
						*pMemOut = fNotMatched;
					}
					pMemOut++;
				}
				pMemIn++;
			}
			if (!(status = os.Write(memout.GetMem(), cNoOfLabelsTmp * LABELS_COUNT * sizeof(n_value), &cbSizeTmp)))
			{
				break;
			}

			cNoOfLabels -= cNoOfLabelsTmp;
		}

		break;
	}

	return status;
}

}//namespace MNIST

}//namespace NeuTron
