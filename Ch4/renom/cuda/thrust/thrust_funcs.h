#ifndef THRUST_FUNCS_H__
#define THRUST_FUNCS_H__
#include "cuda_runtime.h"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>


__device__ VALUE_TYPE atomicAdd(VALUE_TYPE *address, const VALUE_TYPE vlaue);

namespace renom{

	void SET_STREAM_NAME(cudaStream_t stream);
	cudaStream_t GET_STREAM_NAME();

	// Operation
	enum Operation {MUL, ADD, DIV, RDIV, SUB, POW, RPOW};

	// Negate function
	void thrust_negate(VALUE_TYPE *first, VALUE_TYPE *last, VALUE_TYPE *out);

	// Relu Forward function
	struct relu_forward_function;
	void thrust_relu_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Relu Backward function
	struct relu_backward_function;
	void thrust_relu_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Relu6 Forward function
	struct relu6_forward_function;
	void thrust_relu6_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Relu6 Backward function
	struct relu6_backward_function;
	void thrust_relu6_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Sigmoid function
	struct sigmoid_function;
	void thrust_sigmoid(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Hard Sigmoid Forward function
	struct hard_sigmoid_forward_function;
	void thrust_hard_sigmoid_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Hard Sigmoid Backward function
	struct hard_sigmoid_backward_function;
	void thrust_hard_sigmoid_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Tanh function
	struct tanh_function;
	void thrust_tanh(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Hard Tanh Forward function
	struct hard_tanh_forward_function;
	void thrust_hard_tanh_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Hard Tanh Backward function
	struct hard_tanh_backward_function;
	void thrust_hard_tanh_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    const unsigned int RENOM_CUDA_MAX_STRIDES= 5;

    struct binop_strides {
        size_t size;
        size_t result_strides[RENOM_CUDA_MAX_STRIDES]; // Max 5
        size_t lhs_strides[RENOM_CUDA_MAX_STRIDES];
        size_t rhs_strides[RENOM_CUDA_MAX_STRIDES];
    };

    void thrust_add(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_mul(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_sub(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_div(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rdiv(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_pow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rpow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);

    void thrust_add_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_mul_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_sub_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_div_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_rdiv_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_pow_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_rpow_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);

    __global__ void cuda_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock);

    void thrust_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock);

    // Add bias
    void thrust_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a);
    __global__ void cuda_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a);

	struct sign_function;
	void thrust_sign(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // RoI Pooling forward
	__global__ void cuda_forward_roi_pool2d(int N, VALUE_TYPE *x, float spatial_scale, int channels, int height, int width, int outh, int outw, VALUE_TYPE *rois, VALUE_TYPE *z, VALUE_TYPE *argmax_data);
	void thrust_forward_roi_pool2d(int N, VALUE_TYPE *x, float spatial_scale, int channels, int height, int width, int outh, int outw, VALUE_TYPE *rois, VALUE_TYPE *z, VALUE_TYPE *argmax_data);

    __global__ void cuda_backward_roi_pool2d(int N, VALUE_TYPE *du, VALUE_TYPE *argmax, VALUE_TYPE *rois, float spatial_scale,
                                        int batch_N, int channels, int height, int width, int outh,
                                        int outw, VALUE_TYPE *dx);
    void thrust_backward_roi_pool2d(int N, VALUE_TYPE *du, VALUE_TYPE *argmax, VALUE_TYPE *rois, float spatial_scale,
                                        int batch_N, int channels, int height, int width, int outh,
                                        int outw, VALUE_TYPE *dx);

    // Fill
    void thrust_fill(VALUE_TYPE value, VALUE_TYPE *a, int size);

    // Log e function
    struct loge_function;
    void thrust_loge(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Log e function
    struct exp_function;
    void thrust_exp(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // sqrt function
    struct sqrt_function;
    void thrust_sqrt(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Cross entropy
    struct cross_entropy_function;
    void thrust_cross_entropy(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, int size);

    // abs
    struct abs_forward_function;
    void thrust_abs_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);
    struct abs_backward_function;
    void thrust_abs_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // sum
    VALUE_TYPE thrust_all_reduce(VALUE_TYPE* a, int size); // sum up all elements.
    __global__ void cuda_strided_sum(VALUE_TYPE *a, VALUE_TYPE *b, int stride, int axis_size, int step, int size);
    void thrust_strided_reduce(VALUE_TYPE* a, VALUE_TYPE* b, int stride, int axis_size, int step, int size);

    // min
    struct min_function;
    void thrust_min(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // max
    struct max_function;
    void thrust_max(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    const unsigned int RENOM_CUDA_MAX_AXIS= 6;

    struct reduce_shape_infos {
        size_t out_size[RENOM_CUDA_MAX_AXIS];
        size_t in_size[RENOM_CUDA_MAX_AXIS];
        size_t group_size[RENOM_CUDA_MAX_AXIS];
    };


    void thrust_reduce_sum(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos);

    void thrust_reduce_max(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos);

    void thrust_reduce_argmax(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        size_t *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos,
        size_t mod, size_t div);

    void thrust_reduce_min(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos);

    void thrust_reduce_argmin(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        size_t *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos,
        size_t mod, size_t div);

    __global__ void cuda_transpose(size_t size, size_t shapesize,
        VALUE_TYPE *src, const size_t src_strides[16],
        VALUE_TYPE *result, const size_t result_strides[16]);

    void thrust_transpose(
        size_t size, size_t shapesize,
        VALUE_TYPE *src, const size_t src_strides[16],
        VALUE_TYPE *result, const size_t result_strides[16]);



    struct getitem_slice_info {
        long long start, stop;
        long long step;

        long long adv_indexes_len;
        long long *adv_indexes;

        size_t stride, dest_stride;
    };

    struct getitem_slice_infos {
        size_t shape_len;
        getitem_slice_info slice_info[16];
        size_t stride_size;
        size_t strides[16];
        size_t broadcasted_strides[16];
    };

    void thrust_getitem(
        VALUE_TYPE *src,
        VALUE_TYPE *result, size_t result_size,
        getitem_slice_infos *info);

    void thrust_setitem(
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *dest,
        getitem_slice_infos *info);

    __global__ void cuda_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len);
    void thrust_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len);

    struct leaky_relu_forward_function;
    void thrust_leaky_relu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Leaky Relu backward
    struct leaky_relu_backward_function;
    void thrust_leaky_relu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    struct elu_forward_function;
    void thrust_elu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Elu backward
    struct elu_backward_function;
    void thrust_elu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Softsign Forward function
    struct softsign_forward_function;
    void thrust_softsign_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Softsign Backward function
    struct softsign_backward_function;
    void thrust_softsign_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    //Swish forward
    struct swish_forward_function;
    void thrust_swish_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    //Swish backward
    struct swish_backward_function;
    void thrust_swish_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Mish Forward function
	struct mish_forward_function;
	void thrust_mish_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Mish Backward function
	struct hard_sigmoid_backward_function;
	void thrust_mish_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);    

		__global__ void cuda_softplus_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);
		void thrust_softplus_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

		__global__ void cuda_softplus_backward(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *dy, int size);
		void thrust_softplus_backward(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *dy, int size);

    // Lstm forward activation without peep hole
    __global__ void cuda_forward_lstm_activate(int N, int M, VALUE_TYPE *u);
    void thrust_forward_lstm_activate(int N, int M, VALUE_TYPE *u);

    // Lstm forward without peep hole
    __global__ void cuda_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z);
    void thrust_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z);

    // Lstm backward activation without peep hole
    __global__ void cuda_backward_lstm_activate(int N, int M, VALUE_TYPE *u);
    void thrust_backward_lstm_activate(int N, int M, VALUE_TYPE *u);

    // Lstm backward without peep hole
    __global__ void cuda_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps, \
                    VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou);
    void thrust_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps, \
                    VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou);

    // Peephole Lstm forward
    __global__ void cuda_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *pstate, VALUE_TYPE *state, VALUE_TYPE *z);
    void thrust_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *pstate, VALUE_TYPE *state, VALUE_TYPE *z);

    // Peephole Lstm backward
    __global__ void cuda_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, \
            VALUE_TYPE *prefg, VALUE_TYPE *wc, VALUE_TYPE *dy, VALUE_TYPE *drt, \
            VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc);

    void thrust_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, \
            VALUE_TYPE *prefg, VALUE_TYPE *wc, VALUE_TYPE *dy, VALUE_TYPE *drt, \
            VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc);

		// Gru forward function
		__global__ void cuda_forward_gru(int X, int Y, int M, VALUE_TYPE *input, VALUE_TYPE *hminus, VALUE_TYPE *u, VALUE_TYPE *ABC, VALUE_TYPE *h);
		void thrust_forward_gru(int X, int Y, int M, VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, VALUE_TYPE *d, VALUE_TYPE *e);

		// Gru backward function
		__global__ void cuda_backward_gru(int X, int Y, int M, int V, VALUE_TYPE *a, VALUE_TYPE *b, \
			VALUE_TYPE *c, VALUE_TYPE *d, VALUE_TYPE *e, VALUE_TYPE *f, VALUE_TYPE *g, \
			VALUE_TYPE *h, VALUE_TYPE *i);
		__global__ void cuda_db_gru(int H, int W, int M, VALUE_TYPE *yc, VALUE_TYPE *db);
		void thrust_backward_gru(int X, int Y, int M, int V, VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, \
			VALUE_TYPE *d, VALUE_TYPE *e, VALUE_TYPE *f, VALUE_TYPE *g, VALUE_TYPE *h, VALUE_TYPE *i);

    // Binarize
    void thrust_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b);
    __global__ void cuda_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b);

    // Embedding
    void thrust_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);
    __global__ void cuda_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);

    void thrust_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);
    __global__ void cuda_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);

		void thrust_optimizer_sgd(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE momentum, VALUE_TYPE *pdy, VALUE_TYPE *ndy);
		__global__ void cuda_optimizer_sgd(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE momentum, VALUE_TYPE *pdy, VALUE_TYPE *ndy);

		void thrust_optimizer_adagrad(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE *pdy, VALUE_TYPE *ndy, VALUE_TYPE *r);
    __global__ void cuda_optimizer_adagrad(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE *pdy, VALUE_TYPE *ndy, VALUE_TYPE *r);

		void thrust_optimizer_rmsprop(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE eta, VALUE_TYPE *ndy, VALUE_TYPE *r);
    __global__ void cuda_optimizer_rmsprop(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE eta, VALUE_TYPE *ndy, VALUE_TYPE *r);

		void thrust_optimizer_adam(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE gamma_orig, VALUE_TYPE beta, VALUE_TYPE beta_orig, VALUE_TYPE min, bool flug, VALUE_TYPE *u, VALUE_TYPE *r, VALUE_TYPE *ndy);
    __global__ void cuda_optimizer_adam(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE gamma_orig, VALUE_TYPE beta, VALUE_TYPE beta_orig, VALUE_TYPE min, bool flug, VALUE_TYPE *u, VALUE_TYPE *r, VALUE_TYPE *ndy);

		void thrust_optimizer_adadelta(int Elems, VALUE_TYPE decay_rate, VALUE_TYPE epsilon, VALUE_TYPE * previous_squared_gradient, VALUE_TYPE * previous_squared_delta, VALUE_TYPE * dy, VALUE_TYPE * new_dy);
		__global__ void cuda_optimizer_adadelta(int Elems, VALUE_TYPE decay_rate, VALUE_TYPE epsilon, VALUE_TYPE * previous_squared_gradient, VALUE_TYPE * previous_squared_delta, VALUE_TYPE * dy, VALUE_TYPE * new_dy);

		void thrust_optimizer_adamax(int Elems, VALUE_TYPE alpha, VALUE_TYPE epsilon, VALUE_TYPE beta1, VALUE_TYPE running_beta1, VALUE_TYPE beta2, VALUE_TYPE running_beta2, VALUE_TYPE * moment1, VALUE_TYPE * moment2, VALUE_TYPE * dy, VALUE_TYPE * new_dy);
		__global__ void cuda_optimizer_adamax(int Elems, VALUE_TYPE alpha, VALUE_TYPE epsilon, VALUE_TYPE beta1, VALUE_TYPE running_beta1, VALUE_TYPE beta2, VALUE_TYPE running_beta2, VALUE_TYPE * moment1, VALUE_TYPE * moment2, VALUE_TYPE * dy, VALUE_TYPE * new_dy);

    void thrust_get_fg_ary_forward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    __global__ void cuda_get_fg_ary_forward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);

    void thrust_get_fg_ary_backward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    __global__ void cuda_get_fg_ary_backward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);

    void thrust_get_ith_ary_forward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    __global__ void cuda_get_ith_ary_forward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);

    void thrust_get_ith_ary_backward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    __global__ void cuda_get_ith_ary_backward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);


    void thrust_get_nth_ary(int N, int M, int i, int j, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    __global__ void cuda_get_nth_ary(int N, int M, int i, int j, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);

    void thrust_assign_pred_box(int N, int M, VALUE_TYPE *x_ptr, VALUE_TYPE *y_ptr, VALUE_TYPE *h_ptr, VALUE_TYPE *w_ptr, VALUE_TYPE *ary_ptr);
    __global__ void cuda_assign_pred_box(int N, int M, VALUE_TYPE *x_ptr, VALUE_TYPE *y_ptr, VALUE_TYPE *h_ptr, VALUE_TYPE *w_ptr, VALUE_TYPE *ary_ptr);

    void thrust_pred_ctr(int N, int M, VALUE_TYPE *arg_ptr, VALUE_TYPE *length_ptr,VALUE_TYPE *ctr_ptr, VALUE_TYPE *ary_ptr);
    __global__ void cuda_pred_ctr(int N, int M, VALUE_TYPE *arg_ptr, VALUE_TYPE *length_ptr,VALUE_TYPE *ctr_ptr, VALUE_TYPE *ary_ptr);

    void thrust_generate_anchors(int A, int K, int N, VALUE_TYPE *shifts, VALUE_TYPE *ratios_ptr, VALUE_TYPE *scales_ptr, int raio_size, int scale_size, int feat_stride, int base_size, VALUE_TYPE *anchors);
    __global__ void cuda_generate_anchors(int A, int K, int N, VALUE_TYPE *shifts_ptr, VALUE_TYPE *ratios_ptr, VALUE_TYPE *scales_ptr, int ratio_size, int scale_size, int feat_stride, int base_size, VALUE_TYPE *anchors_ptr);

    void thrust_get_ith_bbox(int N, int M, VALUE_TYPE *bbox_ptr, int i, VALUE_TYPE *ary_ptr);
    __global__ void cuda_get_ith_bbox(int N, int M, VALUE_TYPE *bbox_ptr, int i, VALUE_TYPE *ary_ptr);

    void thrust_clip_roi(int N, int M, VALUE_TYPE *roi_ptr, int start, int end, int step, int min_v, int max_v, VALUE_TYPE *ary_ptr);
    __global__ void cuda_clip_roi(int N, int M, VALUE_TYPE *roi_ptr, int start, int end, int step, int min_v, int max_v, VALUE_TYPE *ary_ptr);

		void thrust_clip(int elem, VALUE_TYPE *array, VALUE_TYPE max, VALUE_TYPE min);
		__global__ void cuda_clip(int elem, VALUE_TYPE *array, VALUE_TYPE max, VALUE_TYPE min);
}
#endif
