#pragma once

#ifdef PYBIND11_INCLUDED

#include <pybind11/pybind11.h>

namespace template_tensors::python::pybind11 {

#define ThisType FromBuffer<TElementType, TRank>
#define SuperType IndexedPointerTensor< \
                    ThisType, \
                    TElementType, \
                    template_tensors::Stride<TRank>, \
                    mem::HOST, \
                    dyn_dimseq_t<TRank> \
                  >

template <typename TElementType, metal::int_ TRank>
class FromBuffer : public SuperType
{
private:
  static_assert(std::is_arithmetic<TElementType>::value, "Elementtype is not a valid numpy type");

  pybind11::buffer m_buffer;

  __host__
  FromBuffer(pybind11::buffer buffer, VectorXT<size_t, TRank> strides, VectorXT<size_t, TRank> dims)
    : SuperType(strides, dims)
    , m_buffer(buffer)
  {
  }

public:
  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(reinterpret_cast<typename std::remove_reference<util::copy_qualifiers_t<TElementType, TThisType2>>::type*>(self.m_buffer.get_data()))
  FORWARD_ALL_QUALIFIERS(data, data2)

  template <metal::int_ TIndex>
  __host__
  dim_t getDynDim() const
  {
    return TIndex < TRank ? m_buffer.get_shape()[TIndex] : 1;
  }

  __host__
  dim_t getDynDim(size_t index) const
  {
    return index < TRank ? m_buffer.get_shape()[index] : 1;
  }

  __host__
  ::boost::python::numpy::ndarray& getNumpyArray()
  {
    return m_buffer;
  }

  __host__
  const ::boost::python::numpy::ndarray& getNumpyArray() const
  {
    return m_buffer;
  }

  template <typename TElementType2, metal::int_ TRank2>
  __host__
  friend FromBuffer<TElementType2, TRank2> fromNumpy(::boost::python::numpy::ndarray arr);
};
#undef SuperType
#undef ThisType


template <typename TElementType, metal::int_ TRank>
__host__
FromNdArray<TElementType, TRank> fromNumpy(::boost::python::numpy::ndarray arr)
{
  if (arr.get_nd() != TRank)
  {
    throw template_tensors::python::InvalidNumpyShapeException(arr.get_nd(), TRank);
  }
  if (arr.get_dtype() != ::boost::python::numpy::dtype::get_builtin<TElementType>())
  {
    throw template_tensors::python::InvalidNumpyElementTypeException(arr.get_dtype().get_itemsize(), sizeof(TElementType));
  }
  VectorXT<size_t, TRank> strides = template_tensors::ref<template_tensors::ColMajor, mem::HOST, TRank>(arr.get_strides()) / sizeof(TElementType);
  VectorXT<size_t, TRank> dims = template_tensors::ref<template_tensors::ColMajor, mem::HOST, TRank>(arr.get_shape());

  return FromNdArray<TElementType, TRank>(arr, strides, dims);
}

template <typename TElementType, metal::int_ TRank>
__host__
FromNdArray<TElementType, TRank> fromNumpy(::boost::python::numpy::ndarray arr, template_tensors::VectorXT<size_t, TRank> dims)
{
  FromNdArray<TElementType, TRank> result = fromNumpy<TElementType, TRank>(arr);
  if (!template_tensors::eq(result.template dims<TRank>(), dims))
  {
    throw template_tensors::python::InvalidNumpyShapeException(result.template dims<TRank>(), dims);
  }
  return result;
}

template <metal::int_ TRank2 = DYN, typename TTensorType, metal::int_ TRank = TRank2 == DYN ? non_trivial_dimensions_num_v<TTensorType>::value : TRank2>
__host__
::boost::python::numpy::ndarray toNumpy(TTensorType&& tensor)
{
  ::boost::python::numpy::ndarray arr(::boost::python::numpy::empty(
    jtuple::tuple_apply(functor::make_tuple(), template_tensors::toTuple(tensor.template dims<TRank>())),
    ::boost::python::numpy::dtype::get_builtin<decay_elementtype_t<TTensorType>>()
  ));
  fromNumpy<decay_elementtype_t<TTensorType>, TRank>(arr) = std::forward<TTensorType>(tensor);

  return arr;
}

} // end of ns template_tensors::python::pybind11

#endif
