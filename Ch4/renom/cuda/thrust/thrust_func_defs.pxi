from libcpp cimport bool

cdef extern from "cuda_runtime.h":
  ctypedef struct CUevent_st:
    pass
  ctypedef struct CUstream_st:
    pass

  ctypedef CUevent_st * cudaEvent_t
  ctypedef CUstream_st *  cudaStream_t

cdef extern from * namespace "renom":
    cdef void thrust_negate(VALUE_TYPE* first, VALUE_TYPE *last, VALUE_TYPE *output)
    cdef void thrust_relu_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_relu_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_relu6_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_relu6_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_sigmoid(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_hard_sigmoid_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_hard_sigmoid_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_tanh(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_hard_tanh_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_hard_tanh_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock)
    cdef void thrust_fill(VALUE_TYPE value, VALUE_TYPE *a, int size)
    cdef void thrust_loge(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_exp(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_sqrt(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_sign(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_cross_entropy(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, int size)
    cdef void thrust_abs_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_abs_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef VALUE_TYPE thrust_all_reduce(VALUE_TYPE* a, int size)
    cdef void thrust_strided_reduce(VALUE_TYPE* a, VALUE_TYPE* b, int stride, int axis_size, int step, int size);
    cdef void thrust_create_mask(VALUE_TYPE *a, int size)
    cdef void thrust_min(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_max(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    cdef struct binop_strides:
        size_t size
        size_t result_strides[16]
        size_t lhs_strides[16]
        size_t rhs_strides[16]

    void thrust_add(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_mul(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_sub(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_div(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rdiv(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_pow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rpow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);

    void thrust_mul_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_add_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_sub_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_div_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_rdiv_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_pow_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_rpow_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);

    cdef const unsigned int RENOM_CUDA_MAX_AXIS

    cdef struct reduce_shape_infos:
        size_t out_size[16]
        size_t in_size[16]
        size_t group_size[16]

    cdef void thrust_reduce_sum(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos)

    cdef void thrust_reduce_min(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos)

    cdef void thrust_reduce_argmin(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        size_t *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos,
        size_t mod, size_t div)

    cdef void thrust_reduce_max(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, const size_t src_size,
        VALUE_TYPE *result, const size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos)

    cdef void thrust_reduce_argmax(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, const size_t src_size,
        size_t *result, const size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos,
        size_t mod, size_t div);

    cdef void thrust_transpose(
        size_t size, size_t shapesize,
        VALUE_TYPE *src, const size_t src_strides[16],
        VALUE_TYPE *result, const size_t result_strides[16]);

    cdef void thrust_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len)

    cdef struct getitem_slice_info:
        long long start, stop;
        long long step;
        long long adv_indexes_len;
        long long *adv_indexes;
        size_t stride, dest_stride;

    cdef struct getitem_slice_infos:
        size_t shape_len;
        getitem_slice_info slice_info[16];
        size_t stride_size;
        size_t strides[16];
        size_t broadcasted_strides[16];


    cdef void thrust_getitem(
            VALUE_TYPE *src,
            VALUE_TYPE *result, size_t result_size,
            getitem_slice_infos *info);

    cdef void thrust_setitem(
            VALUE_TYPE *src, size_t src_size,
            VALUE_TYPE *dest,
            getitem_slice_infos *info);

    cdef void thrust_leaky_relu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_leaky_relu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_elu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_elu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_softplus_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_softplus_backward(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *dy, int size);
    cdef void thrust_softsign_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_softsign_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_swish_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_swish_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_mish_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_mish_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_forward_roi_pool2d(int N, VALUE_TYPE *x, float spatial_scale,
                                        int channels, int height, int width, int outh,
                                        int outw, VALUE_TYPE *rois, VALUE_TYPE *z,
                                        VALUE_TYPE *argmax_data);
    cdef void thrust_backward_roi_pool2d(int N, VALUE_TYPE *du, VALUE_TYPE *argmax, VALUE_TYPE *rois, float spatial_scale,
                                        int batch_N, int channels, int height, int width, int outh,
                                        int outw, VALUE_TYPE *dx);
    cdef void thrust_forward_lstm_activate(int N, int M, VALUE_TYPE *u);
    cdef void thrust_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z);
    cdef void thrust_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps,
            VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou);

    cdef void thrust_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *prestate, VALUE_TYPE *state, VALUE_TYPE *z)

    cdef void thrust_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, VALUE_TYPE *prefg, VALUE_TYPE *wc,\
             VALUE_TYPE *dy, VALUE_TYPE *drt, VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc);

    cdef void thrust_forward_gru(int X, int Y, int M, VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, VALUE_TYPE *d, VALUE_TYPE *e);
    cdef void thrust_backward_gru(int X, int Y, int M, int V, VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, VALUE_TYPE *d, \
    VALUE_TYPE *e, VALUE_TYPE *f, VALUE_TYPE *g, VALUE_TYPE *h, VALUE_TYPE *i);

    cdef void thrust_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b);
    cdef void thrust_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);

    cdef void thrust_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);
    cdef void thrust_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a);

    cdef void thrust_optimizer_sgd(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE momentum, VALUE_TYPE *pdy, VALUE_TYPE *ndy);
    cdef void thrust_optimizer_adagrad(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE *pdy, VALUE_TYPE *ndy, VALUE_TYPE *r);
    cdef void thrust_optimizer_rmsprop(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE eta, VALUE_TYPE *ndy, VALUE_TYPE *r);
    cdef void thrust_optimizer_adam(int Elems, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE gamma_orig, VALUE_TYPE beta, VALUE_TYPE beta_orig, VALUE_TYPE min, bool flug, VALUE_TYPE *u, VALUE_TYPE *r, VALUE_TYPE *ndy);
    cdef void thrust_optimizer_adadelta(int Elems, VALUE_TYPE decay_rate, VALUE_TYPE epsilon, VALUE_TYPE * previous_squared_gradient, VALUE_TYPE * previous_squared_delta, VALUE_TYPE * dy, VALUE_TYPE * new_dy);
    cdef void thrust_optimizer_adamax(int Elems, VALUE_TYPE alpha, VALUE_TYPE epsilon, VALUE_TYPE beta1, VALUE_TYPE running_beta1, VALUE_TYPE beta2, VALUE_TYPE running_beta2, VALUE_TYPE * moment1, VALUE_TYPE * moment2, VALUE_TYPE * dy, VALUE_TYPE * new_dy);

    cdef void thrust_get_fg_ary_forward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    cdef void thrust_get_fg_ary_backward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);

    cdef void thrust_get_ith_ary_forward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    cdef void thrust_get_ith_ary_backward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);

    cdef void thrust_get_nth_ary(int N, int M, int i, int j, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2);
    cdef void thrust_assign_pred_box(int N, int M, VALUE_TYPE *x_ptr, VALUE_TYPE *y_ptr,  VALUE_TYPE *h_ptr, VALUE_TYPE *w_ptr, VALUE_TYPE *ary_ptr)
    cdef void thrust_pred_ctr(int N, int M, VALUE_TYPE *arg_ptr, VALUE_TYPE *length_ptr,VALUE_TYPE *ctr_ptr, VALUE_TYPE *ary_ptr)
    cdef void thrust_generate_anchors(int A, int K, int N, VALUE_TYPE *shifts_ptr, VALUE_TYPE *ratios_ptr, VALUE_TYPE *scales_ptr, int ratio_size, int scale_size, int feat_stride, int base_size, VALUE_TYPE *anchors_ptr)

    cdef void thrust_get_ith_bbox(int N, int M, VALUE_TYPE *bbox_ptr, int i, VALUE_TYPE *ary_ptr)
    cdef void thrust_clip_roi(int N, int M, VALUE_TYPE *roi_ptr, int start, int end, int step, int min_v, int max_v, VALUE_TYPE *ary_ptr)
    cdef void thrust_clip(int elem, VALUE_TYPE *array, VALUE_TYPE max, VALUE_TYPE min)
