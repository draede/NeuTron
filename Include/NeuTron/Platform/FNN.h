
#pragma once


#include "NeuTron/Platform/NN.h"


#ifdef __cplusplus
extern "C" {
#endif


#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_mem_size
{
	n_size                       cbTotal;
	n_size                       cbInfo;
	n_size                       cbLayersInfo;
	n_size                       cbLayersData;
	n_size                       cbLinksInfo;
	n_size                       cbLinksData;
	n_size                       cbInputsInfo;
	n_size                       cbOutputsInfo;
}n_fnn_mem_size;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_info
{
	n_layers_count               cLayers;
	n_neurons_count              cInputs;
	n_neurons_count              cOutputs;
	n_value                      fLearningRate;
	n_error_function             nErrorFunction;
	n_uint8                      cErrorArgs;
	n_value                      errorArgs[N_MAX_ERROR_ARGS_COUNT];
	n_fnn_mem_size               memsize;
}n_fnn_info;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_layer_info
{
	n_neurons_count              cNeurons;
	n_layer_flags                nFlags;
	n_activation_function        nActivationFunction;
	n_value                      fBiasValue;
	n_uint8                      cActivationArgs;
	n_value                      activationArgs[N_MAX_ACTIVATION_ARGS_COUNT];
	n_offset                     cbInOffset;
	n_offset                     cbOutOffset;
	n_offset                     cbErrorOffset;
	n_offset                     cbErrorTmpOffset;
	n_offset                     cbBiasesOffset;
	n_offset                     cbBiasesTmpOffset;
}n_fnn_layer_info;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_layers_info
{
	n_layers_count               cCount;
	n_fnn_layer_info             layers[1];
}n_fnn_layers_info;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_layers
{
	n_buffer                     info;
	n_buffer                     data;
	void                         *pCECTX;
}n_fnn_layers;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_link_info
{
	n_neurons_count              cRows;
	n_neurons_count              cColumns;
	n_offset                     cbActualOffset;
	n_offset                     cbAdjustedOffset;
}n_fnn_link_info;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_links_info
{
	n_layers_count               cCount;
	n_fnn_link_info              links[1];
}n_fnn_links_info;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_links
{
	n_buffer                     info;
	n_buffer                     data;
	void                         *pCECTX;
}n_fnn_links;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_inputs_info
{
	n_size                       cBatchSize;
	n_neurons_count              cValues;
}n_fnn_inputs_info;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_inputs
{
	n_buffer                     info;
	n_buffer                     data;
	void                         *pCECTX;
}n_fnn_inputs;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_outputs_info
{
	n_size                       cBatchSize;
	n_neurons_count              cValues;
}n_fnn_outputs_info;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn_outputs
{
	n_buffer                     info;
	n_buffer                     data;
	void                         *pCECTX;
}n_fnn_outputs;
#pragma pack(pop)

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_fnn
{
	n_buffer                     info;
	n_fnn_layers                 layers;
	n_fnn_links                  links;
	n_fnn_inputs                 inputs;
	n_fnn_outputs                outputs;
}n_fnn;
#pragma pack(pop)


#ifdef __cplusplus
}
#endif
