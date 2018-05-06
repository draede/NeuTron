
#pragma once


#include "NeuTron/Platform/NN.h"
#include "CX/Types.hpp"
#include "CX/Status.hpp"
#include "CX/IO/IInputStream.hpp"


namespace NeuTron
{

class DataSet
{
public:

	static const CX::UInt32   FLOAT_FILE_MAGIC  = 0x5344464E; //NFDS - NeuTron DataSet Float
	static const CX::UInt32   DOUBLE_FILE_MAGIC = 0x4444464E; //NFDD - NeuTron DataSet Double

	static const CX::Size     HEADER_SIZE       = sizeof(CX::UInt32) * 3;

	/*
	FORMAT:
	UInt32 = magic (FLOAT_FILE_MAGIC or DOUBLE_FILE_MAGIC)
	UInt32 = dataset entries count
	UInt32 = dataset entry count of values
	*/

	DataSet();

	virtual ~DataSet();

	CX::Status Open(const CX::Char *szInputsPath, const CX::Char *szOutputsPath, CX::Size cbBatchMaxMemSize, 
	                CX::Bool bShuffle = CX::True);

	CX::Status Close();

	CX::Bool IsOK() const;

	const CX::Char *GetInputsPath() const;

	const CX::Char *GetOutputsPath() const;

	CX::Size GetBatchMaxMemSize() const;

	CX::Bool GetShuffle() const;

	n_size GetDataSetEntriesCount() const;

	n_size GetIterationsCount() const;

	n_size GetCurrentIterationIndex() const;

	n_size GetCurrentIterationBatchSize() const;

	n_value *GetInputs() const;

	n_neurons_count GetInputsCount() const;

	n_value *GetOutputs() const;

	n_neurons_count GetOutputsCount() const;

	CX::Status Reset();

	CX::Status Next();

private:

	enum State
	{
		State_None,
		State_Begin,
		State_Running,
		State_End,
	};

	State                  m_nState;
	CX::IO::IInputStream   *m_pInputsIS;
	CX::IO::IInputStream   *m_pOutputsIS;
	CX::String             m_sInputsPath;
	CX::String             m_sOutputsPath;
	CX::Size               m_cbBatchMaxMemSize;
	CX::Bool               m_bShuffle;
	n_size                 m_cDataSetEntries;
	n_size                 m_cIterations;
	n_size                 m_cCurrentIterationIndex;
	n_size                 m_cCurrentIterationBatchSize;
	n_value                *m_pInputs;
	n_neurons_count        m_cInputs;
	CX::Size               m_cbInputsSize;
	n_value                *m_pOutputs;
	n_neurons_count        m_cOutputs;
	CX::Size               m_cbOutputsSize;
	CX::UInt64             m_cbInputsFileSize;
	CX::UInt64             m_cbInputsFileOffset;
	CX::UInt64             m_cbOutputsFileSize;
	CX::UInt64             m_cbOutputsFileOffset;

	void Shuffle(n_value fAmount, n_size cBatchSize, n_value *inputs, n_neurons_count cInputs, 
	             n_value *outputs, n_neurons_count cOutputs);

};

}//namespace NeuTron
