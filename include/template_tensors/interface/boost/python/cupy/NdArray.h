#pragma once

#ifdef BOOST_PYTHON_INCLUDED

#include <boost/python/list.hpp>

namespace template_tensors {

namespace boost {

namespace python {

class InvalidCupyShapeException : public std::exception
{
public:
  InvalidCupyShapeException(size_t got, size_t expected)
    : m_message(std::string("Invalid cupy shape. Got rank ") + std::to_string(got) + " expected rank " + std::to_string(expected))
  {
  }

  template <typename TVector1, typename TVector2>
  InvalidCupyShapeException(TVector1&& got, TVector2&& expected)
    : m_message(std::string("Invalid cupy shape. Got dimensions ") + util::to_string(got) + " expected dimensions " + util::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidCupyElementTypeException : public std::exception
{
public:
  InvalidCupyElementTypeException(std::string got, std::string expected)
    : m_message(std::string("Invalid cupy elementtype. Got size ") + got + " expected size " + expected)
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

namespace detail {

template <typename TElementType>
struct cupy_elementtype_name;

template <>
struct cupy_elementtype_name<float>
{
  static std::string get()
  {
    return "float32";
  }
};

template <>
struct cupy_elementtype_name<double>
{
  static std::string get()
  {
    return "float64";
  }
};

template <>
struct cupy_elementtype_name<uint8_t>
{
  static std::string get()
  {
    return "uint8";
  }
};

template <>
struct cupy_elementtype_name<uint16_t>
{
  static std::string get()
  {
    return "uint16";
  }
};

template <>
struct cupy_elementtype_name<uint32_t>
{
 static std::string get()
  {
    return "uint32";
  }
};

template <>
struct cupy_elementtype_name<uint64_t>
{
  static std::string get()
  {
    return "uint64";
  }
};

template <>
struct cupy_elementtype_name<int8_t>
{
  static std::string get()
  {
    return "int8";
  }
};

template <>
struct cupy_elementtype_name<int16_t>
{
  static std::string get()
  {
    return "int16";
  }
};

template <>
struct cupy_elementtype_name<int32_t>
{
  static std::string get()
  {
    return "int32";
  }
};

template <>
struct cupy_elementtype_name<int64_t>
{
  static std::string get()
  {
    return "int64";
  }
};

} // end of ns detail

template <typename TElementType, metal::int_ TRank>
__host__
auto fromCupy(::boost::python::object arr) -> decltype(template_tensors::ref<mem::DEVICE>(std::declval<TElementType*>(),
  std::declval<template_tensors::Stride<TRank>>(), std::declval<template_tensors::VectorXs<TRank>>()))
{
  TElementType* data_ptr_d = (TElementType*) (size_t) ::boost::python::extract<size_t>(arr.attr("data").attr("ptr"));
  TElementType* data_mem_ptr_d = (TElementType*) (size_t) ::boost::python::extract<size_t>(arr.attr("data").attr("mem").attr("ptr"));
  ASSERT_(data_ptr_d == data_mem_ptr_d, "Invalid cupy interface");

  // size_t itemsize = (size_t) ::boost::python::extract<size_t>(arr.attr("itemsize"));
  if (detail::cupy_elementtype_name<TElementType>::get() != (std::string) ::boost::python::extract<std::string>(arr.attr("dtype").attr("name")))
  {
    throw InvalidCupyElementTypeException((std::string) ::boost::python::extract<std::string>(arr.attr("dtype").attr("name")), detail::cupy_elementtype_name<TElementType>::get());
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

  ::boost::python::object cupy_array;
  {
    template_tensors::boost::python::with_gil guard;
    ::boost::python::object cupy_module = ::boost::python::import("cupy");

    cupy_array = cupy_module.attr("empty")(
      ::tuple::for_all(functor::make_tuple(), template_tensors::toTuple(tensor.template dims<TRank>())),
      detail::cupy_elementtype_name<ElementType>::get(),
      "C"
    );
  }

  fromCupy<ElementType, TRank>(cupy_array) = util::forward<TTensorType>(tensor);

  return cupy_array;
}

} // end of ns python

} // end of ns boost

} // end of ns tensor

#endif
