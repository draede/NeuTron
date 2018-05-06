
#include "NeuTron/DataSet.hpp"
#include "CX/IO/FileInputStream.hpp"
#include "CX/Sys/ByteOrder.hpp"
#include "CX/Mem.hpp"
#include "CX/Util/RndGen.hpp"


using namespace CX;


namespace NeuTron
{

DataSet::DataSet()
{
	m_nState                     = State_None;
	m_pInputsIS                  = NULL;
	m_pOutputsIS                 = NULL;
	m_cbBatchMaxMemSize          = 0;
	m_bShuffle                   = True;
	m_cDataSetEntries            = 0;
	m_cIterations                = 0;
	m_cCurrentIterationIndex     = 0;
	m_cCurrentIterationBatchSize = 0;
	m_pInputs                    = NULL;
	m_cInputs                    = 0;
	m_cbInputsSize               = 0;
	m_pOutputs                   = NULL;
	m_cOutputs                   = 0;
	m_cbOutputsSize              = 0;
	m_cbInputsFileSize           = 0;
	m_cbInputsFileOffset         = 0;
	m_cbOutputsFileSize          = 0;
	m_cbOutputsFileOffset        = 0;
}

DataSet::~DataSet()
{
	Close();
}



Status DataSet::Open(const Char *szInputsPath, const Char *szOutputsPath, Size cbBatchMaxMemSize,
                     Bool bShuffle/* = True*/)
{
	UInt32   nInputsMagic;
	UInt32   cInputsDataSetEntries;
	UInt32   cInputsValues;
	UInt32   nOutputsMagic;
	UInt32   cOutputsDataSetEntries;
	UInt32   cOutputsValues;
	Size     cValues;
	Size     cTotalInputsValues;
	Size     cTotalOutputsValues;
	Size     cbSize;
	Status   status;

	Close();

	for (;;)
	{
		if (NULL == (m_pInputsIS = new IO::FileInputStream(szInputsPath)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		if (!m_pInputsIS->IsOK())
		{
			status = Status_OpenFailed;

			break;
		}
		if (NULL == (m_pOutputsIS = new IO::FileInputStream(szOutputsPath)))
		{
			status = Status_MemAllocFailed;

			break;
		}
		if (!m_pOutputsIS->IsOK())
		{
			status = Status_OpenFailed;

			break;
		}

		if (!(status = m_pInputsIS->Read(&nInputsMagic, sizeof(nInputsMagic), &cbSize)))
		{
			break;
		}
		if (sizeof(nInputsMagic) != cbSize)
		{
			status = Status_ReadFailed;

			break;
		}
		nInputsMagic = Sys::ByteOrder::LE2H(nInputsMagic);
		if (sizeof(n_float) == sizeof(n_value))
		{
			if (FLOAT_FILE_MAGIC != nInputsMagic)
			{
				status = Status(Status_ParseFailed, "Invalid inputs magic");

				break;
			}
		}
		else
		{
			if (DOUBLE_FILE_MAGIC != nInputsMagic)
			{
				status = Status(Status_ParseFailed, "Invalid inputs magic");

				break;
			}
		}
		if (!(status = m_pInputsIS->Read(&cInputsDataSetEntries, sizeof(cInputsDataSetEntries), &cbSize)))
		{
			break;
		}
		if (sizeof(cInputsDataSetEntries) != cbSize)
		{
			status = Status_ReadFailed;

			break;
		}
		cInputsDataSetEntries = Sys::ByteOrder::LE2H(cInputsDataSetEntries);
		if (!(status = m_pInputsIS->Read(&cInputsValues, sizeof(cInputsValues), &cbSize)))
		{
			break;
		}
		if (sizeof(cInputsValues) != cbSize)
		{
			status = Status_ReadFailed;

			break;
		}
		cInputsValues = Sys::ByteOrder::LE2H(cInputsValues);

		if (!(status = m_pOutputsIS->Read(&nOutputsMagic, sizeof(nOutputsMagic), &cbSize)))
		{
			break;
		}
		if (sizeof(nOutputsMagic) != cbSize)
		{
			status = Status_ReadFailed;

			break;
		}
		nOutputsMagic = Sys::ByteOrder::LE2H(nOutputsMagic);
		if (sizeof(n_float) == sizeof(n_value))
		{
			if (FLOAT_FILE_MAGIC != nOutputsMagic)
			{
				status = Status(Status_ParseFailed, "Invalid outputs magic");

				break;
			}
		}
		else
		{
			if (DOUBLE_FILE_MAGIC != nOutputsMagic)
			{
				status = Status(Status_ParseFailed, "Invalid outputs magic");

				break;
			}
		}
		if (!(status = m_pOutputsIS->Read(&cOutputsDataSetEntries, sizeof(cOutputsDataSetEntries), &cbSize)))
		{
			break;
		}
		if (sizeof(cOutputsDataSetEntries) != cbSize)
		{
			status = Status_ReadFailed;

			break;
		}
		cOutputsDataSetEntries = Sys::ByteOrder::LE2H(cOutputsDataSetEntries);
		if (!(status = m_pOutputsIS->Read(&cOutputsValues, sizeof(cOutputsValues), &cbSize)))
		{
			break;
		}
		if (sizeof(cOutputsValues) != cbSize)
		{
			status = Status_ReadFailed;

			break;
		}
		cOutputsValues = Sys::ByteOrder::LE2H(cOutputsValues);

		if (0 == cInputsValues)
		{
			status = Status(Status_ParseFailed, "Invalid inputs values count");

			break;
		}

		if (0 == cOutputsValues)
		{
			status = Status(Status_ParseFailed, "Invalid outputs values count");

			break;
		}

		if (cInputsDataSetEntries != cOutputsDataSetEntries)
		{
			status = Status(Status_ParseFailed, "Mismatched dataset entries count");

			break;
		}

		if (0 == cInputsDataSetEntries)
		{
			status = Status(Status_ParseFailed, "Invalid dataset entries count");

			break;
		}

		if (!(status = m_pInputsIS->GetSize(&m_cbInputsFileSize)))
		{
			break;
		}
		if (!(status = m_pOutputsIS->GetSize(&m_cbOutputsFileSize)))
		{
			break;
		}
		m_cInputs  = cInputsValues;
		m_cOutputs = cOutputsValues;

		if (cbBatchMaxMemSize < cInputsValues * sizeof(n_value) + cOutputsValues * sizeof(n_value))
		{
			status = Status(Status_ParseFailed, "Invalid batch max mem size");

			break;
		}

		cValues = cbBatchMaxMemSize / (cInputsValues * sizeof(n_value) + cOutputsValues * sizeof(n_value));
		if (cValues > cInputsDataSetEntries)
		{
			cValues = cInputsDataSetEntries;
		}

		m_cbInputsSize  = cValues * m_cInputs * sizeof(n_value);
		m_cbOutputsSize = cValues * m_cOutputs * sizeof(n_value);

		if (0 != (m_cbInputsFileSize - HEADER_SIZE) % (cInputsValues * sizeof(n_value)))
		{
			status = Status(Status_ParseFailed, "Mismatched inputs file size");

			break;
		}
		if (0 != (m_cbOutputsFileSize - HEADER_SIZE) % (cOutputsValues * sizeof(n_value)))
		{
			status = Status(Status_ParseFailed, "Mismatched outputs file size");

			break;
		}

		cTotalInputsValues  = (Size)((m_cbInputsFileSize - HEADER_SIZE) / (cInputsValues * sizeof(n_value)));
		cTotalOutputsValues = (Size)((m_cbOutputsFileSize - HEADER_SIZE) / (cOutputsValues * sizeof(n_value)));

		if (cTotalInputsValues != cTotalOutputsValues)
		{
			status = Status(Status_ParseFailed, "Mismatched data set entries count");

			break;
		}

		if (cInputsDataSetEntries != cTotalInputsValues)
		{
			status = Status(Status_ParseFailed, "Mismatched inputs file size");

			break;
		}

		if (cOutputsDataSetEntries != cTotalOutputsValues)
		{
			status = Status(Status_ParseFailed, "Mismatched outputs file size");

			break;
		}

		if (NULL == (m_pInputs = (n_value *)Mem::Alloc(m_cbInputsSize)))
		{
			status = Status_MemAllocFailed;

			break;
		}

		if (NULL == (m_pOutputs = (n_value *)Mem::Alloc(m_cbOutputsSize)))
		{
			status = Status_MemAllocFailed;

			break;
		}

		m_cDataSetEntries            = cTotalInputsValues;
		m_cIterations                = cTotalInputsValues/ cValues;
		if (0 != cTotalInputsValues % cValues)
		{
			m_cIterations++;
		}

		m_nState                     = State_Begin;
		m_sInputsPath                = szInputsPath;
		m_sOutputsPath               = szOutputsPath;
		m_cbBatchMaxMemSize          = cbBatchMaxMemSize;
		m_bShuffle                   = bShuffle;
		m_cCurrentIterationIndex     = 0;
		m_cCurrentIterationBatchSize = 0;
		m_cbInputsFileOffset         = HEADER_SIZE;
		m_cbOutputsFileOffset        = HEADER_SIZE;

		break;
	}
	if (!status)
	{
		Close();
	}

	return status;
}

Status DataSet::Close()
{
	if (NULL != m_pInputs)
	{
		Mem::Free(m_pInputs);
	}
	if (NULL != m_pOutputs)
	{
		Mem::Free(m_pOutputs);
	}
	if (NULL != m_pInputsIS)
	{
		delete m_pInputsIS;
	}
	if (NULL != m_pOutputsIS)
	{
		delete m_pOutputsIS;
	}
	m_nState                     = State_None;
	m_pInputsIS                  = NULL;
	m_pOutputsIS                 = NULL;
	m_sInputsPath.clear();
	m_sOutputsPath.clear();
	m_cbBatchMaxMemSize          = 0;
	m_bShuffle                   = True;
	m_cDataSetEntries            = 0;
	m_cIterations                = 0;
	m_cCurrentIterationIndex     = 0;
	m_cCurrentIterationBatchSize = 0;
	m_pInputs                    = NULL;
	m_cInputs                    = 0;
	m_cbInputsSize               = 0;
	m_pOutputs                   = NULL;
	m_cOutputs                   = 0;
	m_cbOutputsSize              = 0;
	m_cbInputsFileSize           = 0;
	m_cbInputsFileOffset         = 0;
	m_cbOutputsFileSize          = 0;
	m_cbOutputsFileOffset        = 0;

	return Status();
}

Bool DataSet::IsOK() const
{
	return (State_None != m_nState);
}

const Char *DataSet::GetInputsPath() const
{
	return m_sInputsPath.c_str();
}

const Char *DataSet::GetOutputsPath() const
{
	return m_sOutputsPath.c_str();
}

Size DataSet::GetBatchMaxMemSize() const
{
	return m_cbBatchMaxMemSize;
}

Bool DataSet::GetShuffle() const
{
	return m_bShuffle;
}

n_size DataSet::GetDataSetEntriesCount() const
{
	return m_cDataSetEntries;
}

n_size DataSet::GetIterationsCount() const
{
	return m_cIterations;
}

n_size DataSet::GetCurrentIterationIndex() const
{
	return m_cCurrentIterationIndex;
}

n_size DataSet::GetCurrentIterationBatchSize() const
{
	return m_cCurrentIterationBatchSize;
}

n_value *DataSet::GetInputs() const
{
	return m_pInputs;
}

n_neurons_count DataSet::GetInputsCount() const
{
	return m_cInputs;
}

n_value *DataSet::GetOutputs() const
{
	return m_pOutputs;
}

n_neurons_count DataSet::GetOutputsCount() const
{
	return m_cOutputs;
}

Status DataSet::Reset()
{
	Status status;

	if (State_None == m_nState)
	{
		return Status_InvalidCall;
	}
	if (State_Begin == m_nState)
	{
		return Status();
	}
	if (!(status = m_pInputsIS->SetPos(HEADER_SIZE)))
	{
		return status;
	}
	if (!(status = m_pOutputsIS->SetPos(HEADER_SIZE)))
	{
		return status;
	}
	m_cCurrentIterationIndex     = 0;
	m_cCurrentIterationBatchSize = 0;
	m_cbInputsFileOffset         = HEADER_SIZE;
	m_cbOutputsFileOffset        = HEADER_SIZE;
	m_nState                     = State_Begin;

	return Status();
}

Status DataSet::Next()
{
	Size     cbInputsSize;
	Size     cbOutputsSize;
	Size     cbSize;
	Status   status;

	if (State_None == m_nState)
	{
		return Status_InvalidCall;
	}
	if (State_End == m_nState)
	{
		return Status_NoMoreData;
	}
	if (m_cbInputsFileOffset >= m_cbInputsFileSize || m_cbOutputsFileOffset >= m_cbOutputsFileSize)
	{
		m_nState = State_End;

		return Status_NoMoreData;
	}

	if (m_cbInputsFileOffset + m_cbInputsSize <= m_cbInputsFileSize)
	{
		cbInputsSize = m_cbInputsSize;
	}
	else
	{
		cbInputsSize = (Size)(m_cbInputsFileSize - m_cbInputsFileOffset);
	}
	if (m_cbOutputsFileOffset + m_cbOutputsSize <= m_cbOutputsFileSize)
	{
		cbOutputsSize = m_cbOutputsSize;
	}
	else
	{
		cbOutputsSize = (Size)(m_cbOutputsFileSize - m_cbOutputsFileOffset);
	}
	if (cbInputsSize / m_cInputs != cbOutputsSize / m_cOutputs)
	{
		return Status_ReadFailed;
	}
	if (0 == cbInputsSize || 0 == cbOutputsSize)
	{
		m_nState = State_End;

		return Status_NoMoreData;
	}

	if (!(status = m_pInputsIS->Read(m_pInputs, cbInputsSize, &cbSize)))
	{
		return status;
	}
	if (cbSize != cbInputsSize)
	{
		return Status_ReadFailed;
	}

	if (!(status = m_pOutputsIS->Read(m_pOutputs, cbOutputsSize, &cbSize)))
	{
		return status;
	}
	if (cbSize != cbOutputsSize)
	{
		return Status_ReadFailed;
	}

	m_cCurrentIterationBatchSize = cbInputsSize / (m_cInputs * sizeof(n_value));

	if (m_bShuffle)
	{
		Shuffle(N_VAL(0.5), m_cCurrentIterationBatchSize, m_pInputs, m_cInputs, m_pOutputs, m_cOutputs);
	}

	if (State_Begin == m_nState)
	{
		m_nState = State_Running;
	}
	else
	{
		m_cCurrentIterationIndex++;
	}

	return Status();
}

void DataSet::Shuffle(n_value fAmount, n_size cBatchSize, n_value *inputs, n_neurons_count cInputs,
                      n_value *outputs, n_neurons_count cOutputs)
{
	n_size        cCount;
	n_size        cIndex;
	n_size        cIndex1;
	n_size        cIndex2;
	n_value       fTmp;
	FILETIME      ftRandTmp;

	GetSystemTimeAsFileTime(&ftRandTmp);
	Util::RndGen::Get().Seed32(ftRandTmp.dwLowDateTime);
	Util::RndGen::Get().Seed64(((UInt64)ftRandTmp.dwHighDateTime << 32) + ftRandTmp.dwLowDateTime);

	cCount = (n_size)(fAmount * cBatchSize);
	for (cIndex = 0; cIndex < cCount; cIndex++)
	{
		cIndex1 = Util::RndGen::Get().GetUInt64Range(0, cBatchSize - 1);
		cIndex2 = Util::RndGen::Get().GetUInt64Range(0, cBatchSize - 1);
		for (n_neurons_count i = 0; i < cInputs; i++)
		{
			fTmp                            = inputs[cIndex1 * cInputs + i];
			inputs[cIndex1 * cInputs + i]   = inputs[cIndex2 * cInputs + i];
			inputs[cIndex2 * cInputs + i]   = fTmp;
		}
		for (n_neurons_count i = 0; i < cOutputs; i++)
		{
			fTmp                            = outputs[cIndex1 * cOutputs + i];
			outputs[cIndex1 * cOutputs + i] = outputs[cIndex2 * cOutputs + i];
			outputs[cIndex2 * cOutputs + i] = fTmp;
		}
	}
}

}//namespace NeuTron
