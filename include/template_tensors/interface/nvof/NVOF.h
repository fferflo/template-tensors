#pragma once
#ifdef NVOF_INCLUDED

#include <NvOFCuda.h>
#include <sstream>

namespace template_tensors {

namespace nvof {

template <typename TPixel, NV_OF_PERF_LEVEL TPerfLevel = NV_OF_PERF_LEVEL_SLOW>
class Op;

namespace detail {

template <typename TElementType, NV_OF_BUFFER_USAGE TBufferUsage>
class Matrix;

} // end of ns detail

using Output = detail::Matrix<NV_OF_FLOW_VECTOR, NV_OF_BUFFER_USAGE_OUTPUT>;
using Cost = detail::Matrix<uint32_t, NV_OF_BUFFER_USAGE_COST>;
template <typename TPixel>
using Input = detail::Matrix<TPixel, NV_OF_BUFFER_USAGE_INPUT>;

void upsample(Output& output, const Output& input);

namespace detail {

template <typename T>
struct BufferFormat;

template <>
struct BufferFormat<uint8_t>
{
  static const NV_OF_BUFFER_FORMAT value = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
};

template <>
struct BufferFormat<template_tensors::VectorXT<uint8_t, 1>>
{
  static const NV_OF_BUFFER_FORMAT value = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
};

template <>
struct BufferFormat<template_tensors::VectorXT<uint8_t, 4>>
{
  static const NV_OF_BUFFER_FORMAT value = NV_OF_BUFFER_FORMAT_ABGR8;
};

#define ThisType Matrix<TElementType, TBufferUsage>
#define SuperType IndexedPointerTensor< \
                                        ThisType, \
                                        TElementType, \
                                        template_tensors::Stride<2>, \
                                        mem::DEVICE, \
                                        dyn_dimseq_t<2> \
                              >
template <typename TElementType, NV_OF_BUFFER_USAGE TBufferUsage>
class Matrix : public SuperType,
               public StoreDimensions<dyn_dimseq_t<2>>
{
public:
  Matrix(Matrix<TElementType, TBufferUsage>&& other)
    : SuperType(static_cast<SuperType&&>(other))
    , StoreDimensions<dyn_dimseq_t<2>>(static_cast<StoreDimensions<dyn_dimseq_t<2>>&&>(other))
    , m_buffer_obj(util::move(other.m_buffer_obj))
    , m_ptr(util::move(other.m_ptr))
  {
  }

  Matrix(const Matrix<TElementType, TBufferUsage>& other) = delete;

  Matrix(NvOFBufferObj&& buffer_obj, NV_OF_CUDA_BUFFER_STRIDE_INFO stride_info)
    : SuperType(
        template_tensors::Stride<2>(1, stride_info.strideInfo[0].strideXInBytes / sizeof(TElementType)),
        template_tensors::Vector2s(buffer_obj->getWidth(), buffer_obj->getHeight())
      )
    , StoreDimensions<dyn_dimseq_t<2>>(template_tensors::Vector2s(buffer_obj->getWidth(), buffer_obj->getHeight()))
    , m_buffer_obj(util::move(buffer_obj))
    , m_ptr(reinterpret_cast<TElementType*>(static_cast<NvOFBufferCudaDevicePtr*>(m_buffer_obj.get())->getCudaDevicePtr()))
  {
    ASSERT(m_buffer_obj->getElementSize() == sizeof(TElementType), "Invalid element type");
    ASSERT(stride_info.numPlanes == 1, "Invalid number of planes");
    ASSERT(stride_info.strideInfo[0].strideXInBytes % sizeof(TElementType) == 0, "Invalid stride");
  }

  Matrix(NvOFBufferObj&& buffer_obj)
    : Matrix(util::move(buffer_obj), static_cast<NvOFBufferCudaDevicePtr*>(buffer_obj.get())->getStrideInfo())
  {
  }

  template <typename TPixel, NV_OF_PERF_LEVEL TPerfLevel = NV_OF_PERF_LEVEL_SLOW>
  Matrix(Op<TPixel, TPerfLevel>& context)
    : Matrix(util::move(context.m_context->CreateBuffers(TBufferUsage, 1)[0]))
  {
  }

  template <typename TPixel, NV_OF_PERF_LEVEL TPerfLevel = NV_OF_PERF_LEVEL_SLOW>
  Matrix(Op<TPixel, TPerfLevel>& context, template_tensors::Vector2s resolution)
    : Matrix(util::move(context.m_context->CreateBuffers(resolution(0), resolution(1), TBufferUsage, 1)[0]))
  {
  }

  Matrix<TElementType, TBufferUsage>& operator=(Matrix<TElementType, TBufferUsage>&& other)
  {
    static_cast<SuperType&>(*this) = static_cast<SuperType&&>(other);
    m_buffer_obj = util::move(other.m_buffer_obj);
    m_ptr = util::move(other.m_ptr);

    return *this;
  }

  Matrix<TElementType, TBufferUsage>& operator=(const Matrix<TElementType, TBufferUsage>& other) = delete;

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(self.m_ptr)
  FORWARD_ALL_QUALIFIERS(data, data2)

  template <typename TPixel, NV_OF_PERF_LEVEL TPerfLevel>
  friend class template_tensors::nvof::Op;

  friend void ::template_tensors::nvof::upsample(Output& output, const Output& input);

private:
  NvOFBufferObj m_buffer_obj;
  TElementType* m_ptr;
};
#undef SuperType
#undef ThisType

} // end of ns detail

class InvalidGridsizeException : public std::exception
{
public:
  InvalidGridsizeException(size_t got, size_t next_min, bool available)
  {
    std::stringstream str;
    str << "Output grid size " << got << " not supported. ";
    if (available)
    {
      str << "Next higher available grid size is " << next_min << ".";
    }
    else
    {
      str << "No larger grid size is available.";
    }
    m_message = str.str();
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

template <typename TPixel, NV_OF_PERF_LEVEL TPerfLevel>
class Op
{
private:
  NvOFObj m_context;
  template_tensors::Vector2s m_resolution;

  static const NV_OF_CUDA_BUFFER_TYPE BUFFER_TYPE = NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR;

  static CUcontext getCurrentContext()
  {
    CUcontext context = nullptr;
    TT_CUDA_DRIVER_SAFE_CALL(cuCtxGetCurrent(&context));
    return context;
  }

public:
  Op(template_tensors::Vector2s resolution, uint32_t grid_size)
    : Op(resolution, grid_size, getCurrentContext())
  {
  }

  Op(template_tensors::Vector2s resolution, uint32_t grid_size, CUcontext context)
    : m_resolution(resolution)
  {
    m_context = NvOFCuda::Create(context, resolution(0), resolution(1), detail::BufferFormat<TPixel>::value,
      BUFFER_TYPE, BUFFER_TYPE, NV_OF_MODE_OPTICALFLOW, TPerfLevel, nullptr, nullptr); // TODO: last two parameters cuda streams

    if (!m_context->CheckGridSize(grid_size))
    {
      uint32_t next_min_grid_size;
      bool available = m_context->GetNextMinGridSize(grid_size, next_min_grid_size);
      throw InvalidGridsizeException(grid_size, next_min_grid_size, available);
    }
    m_context->Init(grid_size, false);
  }

  Op(template_tensors::Vector2s resolution, CUcontext context)
    : m_resolution(resolution)
  {
    m_context = NvOFCuda::Create(context, resolution(0), resolution(1), detail::BufferFormat<TPixel>::value,
      BUFFER_TYPE, BUFFER_TYPE, NV_OF_MODE_OPTICALFLOW, TPerfLevel, nullptr, nullptr); // TODO: last two parameters cuda streams

    uint32_t next_min_grid_size;
    bool available = m_context->GetNextMinGridSize(0, next_min_grid_size);
    if (!available)
    {
      throw InvalidGridsizeException(0, next_min_grid_size, available);
    }
    m_context->Init(next_min_grid_size, false);
  }

  template_tensors::Vector2s getResolution() const
  {
    return m_resolution;
  }

  void operator()(Output& output, const Input<TPixel>& input1, const Input<TPixel>& input2)
  {
    m_context->Execute(input1.m_buffer_obj.get(), input2.m_buffer_obj.get(), output.m_buffer_obj.get(), nullptr, nullptr, 0, nullptr);
    TT_CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
  }

  void operator()(Output& output, Cost& cost, const Input<TPixel>& input1, const Input<TPixel>& input2)
  {
    m_context->Execute(input1.m_buffer_obj.get(), input2.m_buffer_obj.get(), output.m_buffer_obj.get(), nullptr, cost.m_buffer_obj.get(), 0, nullptr);
    TT_CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
  }

  template <typename TElementType, NV_OF_BUFFER_USAGE TBufferUsage>
  friend class detail::Matrix;
};

void upsample(Output& output, const Output& input)
{
  uint32_t factor = output.rows() / input.rows();
  ASSERT(factor * input.rows() == output.rows() && factor * input.cols() == output.cols(), "Incompatible dimensions");
  NvOFUtilsCuda utils(NV_OF_MODE_OPTICALFLOW);
  utils.Upsample(input.m_buffer_obj.get(), output.m_buffer_obj.get(), factor);
}

__host__ __device__
template_tensors::Vector2f decode_flow(NV_OF_FLOW_VECTOR flow)
{
  return template_tensors::Vector2f(float(flow.flowx) / 32, float(flow.flowy) / 32);
}

namespace functor {

struct decode_flow
{
  __host__ __device__
  template_tensors::Vector2f operator()(NV_OF_FLOW_VECTOR flow) const
  {
    return nvof::decode_flow(flow);
  }
};

} // end of ns functor

} // end of ns nvof

} // end of ns template_tensors

#endif
