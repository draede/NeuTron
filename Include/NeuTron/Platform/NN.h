
#pragma once


#define N_MIN_LAYERS_COUNT                   ((n_layers_count)2)
#define N_MAX_LAYERS_COUNT                   ((n_layers_count)65535)

#define N_MIN_NEURONS_COUNT                  ((n_layers_count)1)
#define N_MAX_NEURONS_COUNT                  ((n_layers_count)65535)

#define N_MAX_ACTIVATION_ARGS_COUNT          ((n_uint8)4)
#define N_MAX_ERROR_ARGS_COUNT               ((n_uint8)4)

//https://en.wikipedia.org/wiki/Activation_function
#define N_ACTIVATION_IDENTITY                ((n_activation_function)1)
#define N_ACTIVATION_SIGMOID                 ((n_activation_function)2)
#define N_ACTIVATION_BINARYSTEP              ((n_activation_function)3)
#define N_ACTIVATION_TANH                    ((n_activation_function)4)
#define N_ACTIVATION_ARCTAN                  ((n_activation_function)5)
#define N_ACTIVATION_SOFTSIGN                ((n_activation_function)6)
#define N_ACTIVATION_ISRU                    ((n_activation_function)7)
#define N_ACTIVATION_RELU                    ((n_activation_function)8)
#define N_ACTIVATION_LEAKYRELU               ((n_activation_function)9)
#define N_ACTIVATION_PRELU                   ((n_activation_function)10)
#define N_ACTIVATION_ELU                     ((n_activation_function)11)
#define N_ACTIVATION_SELU                    ((n_activation_function)12)
#define N_ACTIVATION_SRELU                   ((n_activation_function)13)
#define N_ACTIVATION_ISRLU                   ((n_activation_function)14)
#define N_ACTIVATION_SOFTPLUS                ((n_activation_function)15)
#define N_ACTIVATION_BENTIDENTITY            ((n_activation_function)16)
#define N_ACTIVATION_SOFTEXPONENTIAL         ((n_activation_function)17)
#define N_ACTIVATION_SINUSOID                ((n_activation_function)18)
#define N_ACTIVATION_SINC                    ((n_activation_function)19)
#define N_ACTIVATION_GAUSSIAN                ((n_activation_function)20)
#define N_ACTIVATION_SOFTMAX                 ((n_activation_function)21)

#define N_ERROR_SQUARE                       ((n_error_function)1)
#define N_ERROR_CROSSENTROPY                 ((n_error_function)2)

#define N_LAYER_FLAG_BIAS                    ((n_layer_flag)1)

#define N_VAL(x)                             x ## f


#ifdef __cplusplus
extern "C" {
#endif


#ifdef NN_HOST

#include <inttypes.h>

typedef int8_t                  n_int8;
typedef uint8_t                 n_uint8;
typedef int16_t                 n_int16;
typedef uint16_t                n_uint16;
typedef int32_t                 n_int32;
typedef uint32_t                n_uint32;
typedef int64_t                 n_int64;
typedef uint64_t                n_uint64;
typedef float                   n_float;

#else

#include <CL/cl.h>

typedef cl_char                 n_int8;
typedef cl_uchar                n_uint8;
typedef cl_short                n_int16;
typedef cl_ushort               n_uint16;
typedef cl_int                  n_int32;
typedef cl_uint                 n_uint32;
typedef cl_long                 n_int64;
typedef cl_ulong                n_uint64;
typedef cl_float                n_float;

#endif

typedef n_uint64                n_offset;
typedef n_uint64                n_size;

#pragma pack(push)
#pragma pack(push, 1)
typedef struct _n_buffer
{
	void                         *pBuffer;
	n_size                       cbSize;
}n_buffer;
#pragma pack(pop)

typedef n_float                 n_value;
typedef n_uint16                n_layers_count;
typedef n_uint16                n_neurons_count;
typedef n_uint8                 n_activation_function;
typedef n_uint8                 n_error_function;
typedef n_uint32                n_layer_flags;
typedef n_uint32                n_layer_flag;


#ifdef __cplusplus
}
#endif
