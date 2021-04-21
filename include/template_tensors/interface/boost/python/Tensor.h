#pragma once

#ifdef BOOST_PYTHON_INCLUDED

#define BOOST_PP_SEQ_ENUM_0
#include <boost/preprocessor.hpp>

namespace template_tensors {

namespace boost {

namespace python {

namespace dispatch {

#if defined(DLPACK_INCLUDED) && defined(BOOST_PYTHON_INCLUDED)
#define DLPACK ((FromDlPack<TElementTypes, TRanks, TMemoryTypes>(object)))
#else
#define DLPACK
#endif

// TODO: this should be disabled when mem::HOST not in TMemoryTypes
#if defined(BOOST_NUMPY_INCLUDED)
#define NUMPY ((FromNumpy<TElementTypes, TRanks>(object)))
#else
#define NUMPY
#endif

#if defined(DLPACK_INCLUDED) && defined(BOOST_PYTHON_INCLUDED)
#define TENSORFLOW ((FromTensorflow<TElementTypes, TRanks, TMemoryTypes>(object)))
#else
#define TENSORFLOW
#endif

template <typename TElementTypes, typename TRanks, typename TMemoryTypes>
auto FromTensor(::boost::python::object& object)
RETURN_AUTO(
  ::dispatch::first(BOOST_PP_SEQ_ENUM(DLPACK NUMPY TENSORFLOW))
)

#undef DLPACK
#undef NUMPY
#undef TENSORFLOW

} // end of ns dispatch

} // end of ns python

} // end of ns boost

} // end of ns tensor

#endif
