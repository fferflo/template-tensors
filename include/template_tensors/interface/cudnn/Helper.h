#pragma once

#include "Cudnn.h"

#if defined(__CUDACC__) && defined(CUDNN_INCLUDED)

namespace template_tensors {

namespace op {

class CudnnTensorDescriptor
{
public:
  __host__
  CudnnTensorDescriptor()
  {
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&m_desc));
  }

  __host__
  ~CudnnTensorDescriptor()
  {
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(m_desc));
  }

  template <typename TDataElementType, typename TVectorType>
  __host__
  void set(const TVectorType& dimensions)
  {
    const metal::int_ RANK = rows_v<TVectorType>::value;
    static_assert(RANK <= CUDNN_DIM_MAX, "Ranks greater than CUDNN_DIM_MAX are not allowed");
    static_assert(RANK >= 4, "Ranks smaller than 4 are not allowed");

    VectorXT<int, RANK> int_dims = dimensions;
    CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptorEx(
      m_desc,
      CUDNN_TENSOR_NCHW,
      cudnn::CudnnDataType<TDataElementType>::value,
      RANK,
      int_dims.data()
    ));
  }

  const cudnnTensorDescriptor_t& getDesc() const
  {
    return m_desc;
  }

private:
  cudnnTensorDescriptor_t m_desc;
};

// Dims: [OutputChannel, InputChannel, Image(row-major)...]
class CudnnFilterDescriptor
{
public:
  __host__
  CudnnFilterDescriptor()
  {
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&m_desc));
  }

  __host__
  ~CudnnFilterDescriptor()
  {
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(m_desc));
  }

  template <typename TDataElementType, typename TVectorType>
  __host__
  void set(const TVectorType& out_channels_in_channels_dimensions)
  {
    const metal::int_ RANK = rows_v<TVectorType>::value;
    static_assert(RANK <= CUDNN_DIM_MAX, "Ranks greater than CUDNN_DIM_MAX are not allowed");
    static_assert(RANK >= 4, "Ranks smaller than 4 are not allowed");

    VectorXT<int, RANK> int_dims = out_channels_in_channels_dimensions;
    CUDNN_SAFE_CALL(cudnnSetFilterNdDescriptor(
      m_desc,
      cudnn::CudnnDataType<TDataElementType>::value,
      CUDNN_TENSOR_NCHW,
      RANK,
      int_dims.data()
    ));
  }

  template <typename TDataElementType, typename TVectorType>
  __host__
  void set(size_t out_channels, size_t in_channels, const TVectorType& dimensions)
  {
    const metal::int_ RANK = rows_v<TVectorType>::value;
    VectorXs<RANK + 2> out_channels_in_channels_dimensions;
    out_channels_in_channels_dimensions(0) = out_channels;
    out_channels_in_channels_dimensions(1) = in_channels;
    template_tensors::tail<RANK>(out_channels_in_channels_dimensions) = dimensions;

    set<TDataElementType>(out_channels_in_channels_dimensions);
  }

  const cudnnFilterDescriptor_t& getDesc() const
  {
    return m_desc;
  }

private:
  cudnnFilterDescriptor_t m_desc;
};

class CudnnConvolutionDescriptor
{
public:
  __host__
  CudnnConvolutionDescriptor()
  {
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&m_desc));
  }

  __host__
  ~CudnnConvolutionDescriptor()
  {
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(m_desc));
  }

  // Dims: [OutputChannel, InputChannel, Image(row-major)...]
  template <typename TElementType, metal::int_ TRank>
  __host__
  void set(
    const VectorXs<TRank>& pad,
    const VectorXs<TRank>& stride,
    const VectorXs<TRank>& dilate,
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION)
  {
    static_assert(TRank >= 2, "Invalid rank given");

    VectorXT<int, TRank> int_pad = pad;
    VectorXT<int, TRank> int_stride = stride;
    VectorXT<int, TRank> int_dilate = dilate;
    ASSERT(m_desc != nullptr, "cudnnConvolutionDescriptor_t cannot be null");
    ASSERT(template_tensors::all(int_pad >= 0), "Padding cannot be strictly negative");
    ASSERT(template_tensors::all(int_stride > 0), "Striding cannot be negative or zero");
    ASSERT(template_tensors::all(int_dilate > 0), "Dilation cannot be negative or zero");
    CUDNN_SAFE_CALL(cudnnSetConvolutionNdDescriptor(
      m_desc,
      TRank,
      int_pad.data(),
      int_stride.data(),
      int_dilate.data(),
      mode,
      cudnn::CudnnDataType<TElementType>::value
    ));
  }

  const cudnnConvolutionDescriptor_t& getDesc() const
  {
    return m_desc;
  }

private:
  cudnnConvolutionDescriptor_t m_desc;
};

// TODO: where to put this
template <typename TAllocator>
class Workspace
{
public:
  Workspace()
    : m_workspace(0)
    , m_size(0)
  {
  }

  void require(size_t size)
  {
    m_size = math::max(m_size, size);
  }

  void ensure()
  {
    if (m_workspace.size() < m_size)
    {
      m_workspace = ::array::AllocArray<uint8_t, mem::alloc::device>(m_size);
    }
  }

  ::array::AllocArray<uint8_t, mem::alloc::device>& get()
  {
    ensure();
    return m_workspace;
  }

  const ::array::AllocArray<uint8_t, mem::alloc::device>& get() const
  {
    ensure();
    return m_workspace;
  }

  size_t size() const
  {
    return m_size;
  }

private:
  ::array::AllocArray<uint8_t, mem::alloc::device> m_workspace;
  size_t m_size;
};

} // end of ns op

} // end of ns tensor

#endif
