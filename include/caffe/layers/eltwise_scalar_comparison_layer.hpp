#ifndef CAFFE_ELTWISE_SCALAR_COMPARISON_LAYER_HPP_
#define CAFFE_ELTWISE_SCALAR_COMPARISON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Tests whether the input is equal to a scalar value: outputs 1 for inputs
 *        equal to value; 0 otherwise.
 */
template <typename Dtype>
class EltwiseScalarComparisonLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides EltwiseScalarComparisonParameter eltwise_scalarcomp_param,
   *     with EltwiseScalarComparisonLayer options:
   *   - value \b
   *     the scalar value @f$ v @f$ to which the input values are compared.
   */
  explicit EltwiseScalarComparisonLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EltwiseScalarComparison"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *       y = \left\{
   *       \begin{array}{lr}
   *         0 & \mathrm{if} \; x \le t \\
   *         1 & \mathrm{if} \; x == t
   *       \end{array} \right.
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);  
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  vector<Dtype> value_;
};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_SCALAR_COMPARISON_LAYER_HPP_
