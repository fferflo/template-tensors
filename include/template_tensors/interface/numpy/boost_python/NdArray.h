#pragma once

#ifdef BOOST_NUMPY_INCLUDED

#include <jtuple/tuple_utility.hpp>
#include <boost/python/list.hpp>
#include <boost/python/numpy.hpp>

namespace template_tensors::python::boost {

#define ThisType FromNdArray<TElementType, TRank>
#define SuperType IndexedPointerTensor< \
                    ThisType, \
                    TElementType, \
                    template_tensors::Stride<TRank>, \
                    mem::HOST, \
                    dyn_dimseq_t<TRank> \
                  >

template <typename TElementType, metal::int_ TRank>
class FromNdArray : public SuperType
                  , public StoreDimensions<dyn_dimseq_t<TRank>>
{
private:
  static_assert(std::is_arithmetic<TElementType>::value, "Elementtype is not a valid numpy type");

  ::boost::python::numpy::ndarray m_numpy;

  __host__
  FromNdArray(::boost::python::numpy::ndarray numpy, VectorXT<size_t, TRank> strides, VectorXT<size_t, TRank> dims)
    : SuperType(strides, dims)
    , StoreDimensions<dyn_dimseq_t<TRank>>(dims)
    , m_numpy(numpy)
  {
  }

public:
  TT_TENSOR_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(reinterpret_cast<typename std::remove_reference<util::copy_qualifiers_t<TElementType, TThisType2>>::type*>(self.m_numpy.get_data()))
  FORWARD_ALL_QUALIFIERS(data, data2)

  __host__
  ::boost::python::numpy::ndarray& getNumpyArray()
  {
    return m_numpy;
  }

  __host__
  const ::boost::python::numpy::ndarray& getNumpyArray() const
  {
    return m_numpy;
  }

  template <typename TElementType2, metal::int_ TRank2>
  __host__
  friend FromNdArray<TElementType2, TRank2> fromNumpy(::boost::python::numpy::ndarray arr);
};
#undef SuperType
#undef ThisType


template <typename TElementType, metal::int_ TRank>
__host__
FromNdArray<TElementType, TRank> fromNumpy(::boost::python::numpy::ndarray arr)
{
  VectorXT<size_t, TRank> strides, dims;
  {
    template_tensors::python::with_gil guard;
    if (arr.get_nd() != TRank)
    {
      throw template_tensors::python::InvalidNumpyShapeException(arr.get_nd(), TRank);
    }
    if (arr.get_dtype() != ::boost::python::numpy::dtype::get_builtin<TElementType>())
    {
      throw template_tensors::python::InvalidNumpyElementTypeException(arr.get_dtype().get_itemsize(), sizeof(TElementType));
    }
    strides = template_tensors::ref<template_tensors::ColMajor, mem::HOST, TRank>(arr.get_strides()) / sizeof(TElementType);
    dims = template_tensors::ref<template_tensors::ColMajor, mem::HOST, TRank>(arr.get_shape());
  }

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
  std::unique_ptr<::boost::python::numpy::ndarray> arr;
  {
    template_tensors::python::with_gil guard;
    arr = std::make_unique<::boost::python::numpy::ndarray>(::boost::python::numpy::empty(
      jtuple::tuple_apply(functor::make_tuple(), template_tensors::toTuple(tensor.template dims<TRank>())),
      ::boost::python::numpy::dtype::get_builtin<decay_elementtype_t<TTensorType>>()
    ));
  }

  fromNumpy<decay_elementtype_t<TTensorType>, TRank>(*arr) = std::forward<TTensorType>(tensor);
  return *arr;
}

} // end of ns template_tensors::python::boost

#endif
