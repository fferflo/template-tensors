#include "Helper.h"

#if defined(__CUDACC__) && defined(CUDNN_INCLUDED)

namespace template_tensors {

namespace op {

class CudnnConvolutionForwardAlgo
{
public:
  static const cudnnConvolutionFwdAlgo_t NO_ALGO = static_cast<cudnnConvolutionFwdAlgo_t>(-1);

  CudnnConvolutionForwardAlgo(cudnnConvolutionFwdAlgo_t algo)
    : m_algo(algo)
  {
  }

  cudnnConvolutionFwdAlgo_t get(
    const CudnnTensorDescriptor& in_desc,
    const CudnnFilterDescriptor& filter_desc,
    const CudnnConvolutionDescriptor& convolution_desc,
    const CudnnTensorDescriptor& out_desc)
  {
    if (m_algo != NO_ALGO)
    {
      return m_algo;
    }
    else
    {
      cudnnConvolutionFwdAlgoPerf_t algo_perf;
      int returned_algo_count;
      CUDNN_SAFE_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
          cudnn::getContext().getHandle(),
          in_desc.getDesc(),
          filter_desc.getDesc(),
          convolution_desc.getDesc(),
          out_desc.getDesc(),
          1,
          &returned_algo_count,
          &algo_perf));
      ASSERT(returned_algo_count == 1, "No convolution algorithm found");
      return algo_perf.algo;
    }
  }

private:
  cudnnConvolutionFwdAlgo_t m_algo;
};



// in/ out dims: [Batch, Channel, Image(row-major)...]
template <typename TDataElementType, metal::int_ TRank, typename TCalculateElementType = TDataElementType>
class CudnnConvolutionForward
{
public:
  __host__
  CudnnConvolutionForward(
        const VectorXs<TRank>& in_dims,
        const VectorXs<TRank>& filter_dims,
        size_t batch_size,
        size_t in_channels,
        size_t out_channels,
        CudnnConvolutionForwardAlgo algo = CudnnConvolutionForwardAlgo::NO_ALGO,
        std::shared_ptr<Workspace<mem::alloc::device>> workspace = std::make_shared<Workspace<mem::alloc::device>>(),
        const VectorXs<TRank>& pad = VectorXs<TRank>(0),
        const VectorXs<TRank>& stride = VectorXs<TRank>(1),
        const VectorXs<TRank>& dilate = VectorXs<TRank>(1),
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION)
    : m_in_dims(in_dims)
    , m_filter_dims(filter_dims)
    , m_out_dims(1 + (m_in_dims + 2 * pad - (((m_filter_dims - 1) * dilate) + 1)) / stride)
    , m_batch_size(batch_size)
    , m_in_channels(in_channels)
    , m_out_channels(out_channels)
    , m_algo()
    , m_workspace(workspace)
  {
    m_in_desc.set<TDataElementType>(template_tensors::concat<0>(Vector1s(batch_size), Vector1s(in_channels), in_dims));
    m_out_desc.set<TDataElementType>(template_tensors::concat<0>(Vector1s(batch_size), Vector1s(out_channels), m_out_dims));
    m_filter_desc.set<TDataElementType>(template_tensors::concat<0>(Vector1s(out_channels), Vector1s(in_channels), filter_dims));
    m_convolution_desc.set<TCalculateElementType>(pad, stride, dilate, mode);

    m_algo = algo.get(m_in_desc, m_filter_desc, m_convolution_desc, m_out_desc);

    size_t workspace_bytes = 0;
    CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn::getContext().getHandle(),
        m_in_desc.getDesc(),
        m_filter_desc.getDesc(),
        m_convolution_desc.getDesc(),
        m_out_desc.getDesc(),
        m_algo,
        &workspace_bytes));
    m_workspace->require(workspace_bytes);
  }

  template <typename TTensorType1, typename TTensorType2, typename TTensorType3>
  __host__
  CudnnConvolutionForward(
        const TTensorType1& in, const TTensorType2& filter, const TTensorType3& out,
        CudnnConvolutionForwardAlgo algo = CudnnConvolutionForwardAlgo::NO_ALGO,
        std::shared_ptr<Workspace<mem::alloc::device>> workspace = std::make_shared<Workspace<mem::alloc::device>>(),
        const VectorXs<TRank>& pad = VectorXs<TRank>(0),
        const VectorXs<TRank>& stride = VectorXs<TRank>(1),
        const VectorXs<TRank>& dilate = VectorXs<TRank>(1),
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION)
    : CudnnConvolutionForward(
        template_tensors::tail<TRank>(in.template dims<TRank + 2>()),
        template_tensors::tail<TRank>(filter.template dims<TRank + 2>()),
        in.template dim<0>(),
        in.template dim<1>(),
        out.template dim<1>(),
        algo,
        workspace,
        pad, stride, dilate,
        mode
      )
  {
    static_assert(std::is_same<decay_elementtype_t<TTensorType1>, TDataElementType>::value, "Invalid element type");
    static_assert(std::is_same<decay_elementtype_t<TTensorType2>, TDataElementType>::value, "Invalid element type");
    static_assert(std::is_same<decay_elementtype_t<TTensorType3>, TDataElementType>::value, "Invalid element type");
    static_assert(mem::isOnDevice<mem::memorytype_v<TTensorType1>::value>() && mem::isOnDevice<mem::memorytype_v<TTensorType2>::value>() && mem::isOnDevice<mem::memorytype_v<TTensorType3>::value>(), "Invalid memory types");

    ASSERT(in.template dim<0>() == out.template dim<0>(), "Batch sizes do not match");
    ASSERT(in.template dim<1>() == filter.template dim<1>(), "Number of channels do not match");
    ASSERT(out.template dim<1>() == filter.template dim<0>(), "Number of channels do not match");
    ASSERT(template_tensors::all(template_tensors::tail<TRank>(out.template dims<TRank + 2>()) == m_out_dims), "Invalid convolution output size");
  }

  // out = alpha * conv(in, filter) + beta * out
  template <typename TTensorType1, typename TTensorType2, typename TTensorType3>
  __host__
  void forward(TCalculateElementType alpha, const TTensorType1& in, const TTensorType2& filter, TCalculateElementType beta, TTensorType3&& out)
  { // TODO: what about index strategy?
    ASSERT(in.template dim<0>() == m_batch_size, "Batch size does not match");
    ASSERT(in.template dim<1>() == m_in_channels, "Number of channels does not match");
    ASSERT(template_tensors::all(template_tensors::tail<TRank>(in.template dims<TRank + 2>()) == m_in_dims), "Convolution input size does not match");
    ASSERT(filter.template dim<0>() == m_out_channels, "Number of channels does not match");
    ASSERT(filter.template dim<1>() == m_in_channels, "Number of channels does not match");
    ASSERT(template_tensors::all(template_tensors::tail<TRank>(filter.template dims<TRank + 2>()) == m_filter_dims), "Convolution filter size does not match");
    ASSERT(out.template dim<0>() == m_batch_size, "Batch size does not match");
    ASSERT(out.template dim<1>() == m_out_channels, "Number of channels does not match");
    ASSERT(template_tensors::all(template_tensors::tail<TRank>(out.template dims<TRank + 2>()) == m_out_dims), "Convolution output size does not match");

    static_assert(mem::isOnDevice<mem::memorytype_v<TTensorType1>::value>() && mem::isOnDevice<mem::memorytype_v<TTensorType2>::value>() && mem::isOnDevice<mem::memorytype_v<TTensorType3>::value>(), "Invalid memory types");

    CUDNN_SAFE_CALL(cudnnConvolutionForward(
      cudnn::getContext().getHandle(),
      &alpha,
      m_in_desc.getDesc(),
      in.data(),
      m_filter_desc.getDesc(),
      filter.data(),
      m_convolution_desc.getDesc(),
      m_algo,
      m_workspace->get().data(),
      m_workspace->size(),
      &beta,
      m_out_desc.getDesc(),
      out.data()));
  }

  const cudnnConvolutionFwdAlgo_t& getAlgorithm() const
  {
    return m_algo;
  }

private:
  VectorXs<TRank> m_in_dims;
  VectorXs<TRank> m_filter_dims;
  VectorXs<TRank> m_out_dims;
  size_t m_batch_size;
  size_t m_in_channels;
  size_t m_out_channels;
  CudnnTensorDescriptor m_in_desc;
  CudnnFilterDescriptor m_filter_desc;
  CudnnTensorDescriptor m_out_desc;
  CudnnConvolutionDescriptor m_convolution_desc;
  cudnnConvolutionFwdAlgo_t m_algo;
  std::shared_ptr<Workspace<mem::alloc::device>> m_workspace;
};

} // end of ns op

} // end of ns template_tensors

#endif
