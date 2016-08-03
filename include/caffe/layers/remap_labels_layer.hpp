#ifndef CAFFE_REMAP_LABELS_LAYER_HPP_
#define CAFFE_REMAP_LABELS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Remap the input label map to a new label map
 */
template <typename Dtype>
class RemapLabelsLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides RemapLabelsParameter remap_labels_param,
   *     with RemapLabelsLayer options:
   *   - oldlabel \b
   *   - newlabel \b
   */
  explicit RemapLabelsLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RemapLabels"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);  
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  vector<Dtype> oldlabel_;
  vector<Dtype> newlabel_;
};

}  // namespace caffe

#endif  // CAFFE_REMAP_LABELS_LAYER_HPP_
