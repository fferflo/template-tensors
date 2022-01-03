#pragma once

#ifdef BOOST_PYTHON_INCLUDED

#include <boost/python/list.hpp>
#include <jtuple/tuple_utility.hpp>

namespace template_tensors::python::boost {

// TODO: should return a class that holds a reference to the python object, so that it is not destructed

template <typename TElementType, metal::int_ TRank>
__host__
auto fromCupy(::boost::python::object arr) -> decltype(template_tensors::ref<mem::DEVICE>(std::declval<TElementType*>(),
  std::declval<template_tensors::Stride<TRank>>(), std::declval<template_tensors::VectorXs<TRank>>()))
{
  TElementType* data_ptr_d = (TElementType*) (size_t) ::boost::python::extract<size_t>(arr.attr("data").attr("ptr"));
  TElementType* data_mem_ptr_d = (TElementType*) (size_t) ::boost::python::extract<size_t>(arr.attr("data").attr("mem").attr("ptr"));
  ASSERT_(data_ptr_d == data_mem_ptr_d, "Invalid cupy interface");

  // size_t itemsize = (size_t) ::boost::python::extract<size_t>(arr.attr("itemsize"));
  if (template_tensors::python::cupy_elementtype_name<TElementType>::get() != (std::string) ::boost::python::extract<std::string>(arr.attr("dtype").attr("name")))
  {
    throw InvalidCupyElementTypeException((std::string) ::boost::python::extract<std::string>(arr.attr("dtype").attr("name")), template_tensors::python::cupy_elementtype_name<TElementType>::get());
  }

  size_t rank = (size_t) ::boost::python::extract<size_t>(arr.attr("ndim"));
  if (rank != TRank)
  {
    throw InvalidCupyShapeException(rank, TRank);
  }

  template_tensors::VectorXs<TRank> shape, strides;
  for (auto i = 0; i < TRank; i++)
  {
    shape(i) = ::boost::python::extract<size_t>(static_cast<::boost::python::tuple>(arr.attr("shape"))[i]);
    strides(i) = ::boost::python::extract<size_t>(static_cast<::boost::python::tuple>(arr.attr("strides"))[i]) / sizeof(TElementType);
  }

  return ref<mem::DEVICE>(data_ptr_d, template_tensors::Stride<TRank>(strides), shape);
}

template <metal::int_ TRank2 = DYN, typename TTensorType, metal::int_ TRank = TRank2 == DYN ? non_trivial_dimensions_num_v<TTensorType>::value : TRank2>
__host__
::boost::python::object toCupy(TTensorType&& tensor)
{
  using ElementType = template_tensors::decay_elementtype_t<TTensorType>;

  ::boost::python::object cupy_module = ::boost::python::import("cupy");
  ::boost::python::object cupy_array = cupy_module.attr("empty")(
    jtuple::tuple_apply(functor::make_tuple(), template_tensors::toTuple(tensor.template dims<TRank>())),
    template_tensors::python::cupy_elementtype_name<ElementType>::get(),
    "C"
  );

  fromCupy<ElementType, TRank>(cupy_array) = std::forward<TTensorType>(tensor);

  return cupy_array;
}

} // end of ns template_tensors::python::boost

#endif
