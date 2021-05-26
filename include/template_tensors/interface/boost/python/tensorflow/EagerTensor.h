#pragma once

#if defined(DLPACK_INCLUDED) && defined(BOOST_PYTHON_INCLUDED)

namespace template_tensors {

namespace boost {

namespace python {

template <typename TElementType, metal::int_ TRank, mem::MemoryType TMemoryType>
__host__
auto fromTensorflow(::boost::python::object tensorflow)
-> decltype(template_tensors::fromDlPack<TElementType, TRank, TMemoryType>(std::declval<SafeDLManagedTensor>()))
{
  ::boost::python::object dlpack;
  {
    template_tensors::boost::python::with_gil guard;
    ::boost::python::object tensorflow_module = ::boost::python::import("tensorflow");
    dlpack = tensorflow_module.attr("experimental").attr("dlpack").attr("to_dlpack")(tensorflow);
  }
  return template_tensors::fromDlPack<TElementType, TRank, TMemoryType>(template_tensors::boost::python::toDlPack(dlpack, "dltensor"));
}

template <metal::int_ TRank2 = DYN, typename TTensorType, metal::int_ TRank = TRank2 == DYN ? non_trivial_dimensions_num_v<TTensorType>::value : TRank2>
__host__
::boost::python::object toTensorflow(TTensorType&& tensor)
{
  ::boost::python::object dlpack = template_tensors::boost::python::fromDlPack(template_tensors::toDlPack(std::forward<TTensorType>(tensor)), "dltensor");
  ::boost::python::object tensorflow;
  {
    template_tensors::boost::python::with_gil guard;
    ::boost::python::object tensorflow_module = ::boost::python::import("tensorflow");
    tensorflow = tensorflow_module.attr("experimental").attr("dlpack").attr("from_dlpack")(dlpack);
  }
  return tensorflow;
}

} // end of ns python

} // end of ns boost

} // end of ns template_tensors

#endif
