#ifdef DLPACK_INCLUDED
#ifdef BOOST_PYTHON_INCLUDED

#include <boost/algorithm/string/join.hpp>

namespace template_tensors {

namespace boost {

namespace python {

namespace dispatch {

template <typename TElementTypes, typename TRanks, typename TMemoryTypes>
struct FromDlPack
{
  struct ElementTypeMatches
  {
    template_tensors::SafeDLManagedTensor& dl;

    template <typename TElementType>
    bool operator()()
    {
      return dl->dl_tensor.dtype.code == template_tensors::dlpack_elementtype<TElementType>::getCode()
          && dl->dl_tensor.dtype.bits == template_tensors::dlpack_elementtype<TElementType>::getBits()
          && dl->dl_tensor.dtype.lanes == template_tensors::dlpack_elementtype<TElementType>::getLanes();
    }
  };

  struct RankMatches
  {
    template_tensors::SafeDLManagedTensor& dl;

    template <metal::int_ TRank>
    bool operator()()
    {
      return dl->dl_tensor.ndim == TRank;
    }
  };

  struct MemoryTypeMatches
  {
    template_tensors::SafeDLManagedTensor& dl;

    template <mem::MemoryType TMemoryType>
    bool operator()()
    {
      return dl->dl_tensor.device.device_type == template_tensors::dlpack_devicetype<TMemoryType>::value;
    }
  };

  static constexpr const char* const ELEMENT_TYPE_STR = "Element type";
  static constexpr const char* const RANK_STR = "Rank";
  static constexpr const char* const MEMORY_TYPE_STR = "Memory type";

  static auto inner_dispatcher(template_tensors::SafeDLManagedTensor& dl)
  RETURN_AUTO(
    ::dispatch::all(
      ::dispatch::first_type<TElementTypes>(ElementTypeMatches{dl}, ELEMENT_TYPE_STR),
      ::dispatch::first_value<metal::int_, TRanks>(RankMatches{dl}, RANK_STR),
      ::dispatch::first_value<mem::MemoryType, TMemoryTypes>(MemoryTypeMatches{dl}, MEMORY_TYPE_STR)
    )
  )

  using InnerDispatcher = decltype(inner_dispatcher(std::declval<template_tensors::SafeDLManagedTensor&>()));

  ::boost::python::object& object;
  std::vector<std::string> names;

  FromDlPack(::boost::python::object& object, std::vector<std::string> names)
    : object(object)
    , names(names)
  {
  }

  template <typename... TNames>
  FromDlPack(::boost::python::object& object, TNames... names)
    : object(object)
    , names{names...}
  {
  }

  struct Result
  {
    std::string message = "";
    typename InnerDispatcher::Result inner_result;

    operator bool() const
    {
      return message.empty() && inner_result;
    }

    std::string error(std::string prefix = "") const
    {
      if (!message.empty())
      {
        return prefix + "DlPack: " + message;
      }
      else
      {
        return prefix + "DlPack:\n" + inner_result.error(prefix + "  ");
      }
    }
  };

  template <typename TFunctor>
  struct Forward
  {
    template_tensors::SafeDLManagedTensor& dl;
    TFunctor functor;

    template <typename TElementType, metal::int_ TRank, mem::MemoryType TMemoryType>
    void operator()(metal::value<TElementType>, std::integral_constant<metal::int_, TRank>, std::integral_constant<mem::MemoryType, TMemoryType>)
    {
      functor(template_tensors::fromDlPack<TElementType, TRank, TMemoryType>(util::move(dl)));
    }
  };

  template <typename TFunctor>
  Result operator()(TFunctor&& functor)
  {
    Result result;
    std::string class_name = template_tensors::boost::python::getClassName(object);
    if (class_name == "PyCapsule")
    {
      std::string name;
      {
        template_tensors::boost::python::with_gil guard;
        name = std::string(PyCapsule_GetName(object.ptr()));
      }
      if (names.empty() || std::find(names.begin(), names.end(), name) != names.end())
      {
        template_tensors::SafeDLManagedTensor dl = template_tensors::boost::python::toDlPack(object, name.c_str());
        result.inner_result = inner_dispatcher(dl)(Forward<TFunctor&&>{dl, util::forward<TFunctor>(functor)});
      }
      else
      {
        result.message = std::string("Capsule name '") + name + "' not in list of allowed names ("
          + ::boost::algorithm::join(names, ", ") + ")";
      }
    }
    else
    {
      result.message = std::string("'") + class_name + "' is not a python capsule";
    }
    return result;
  }
};

} // end of ns dispatch

} // end of ns python

} // end of ns boost

} // end of ns template_tensors

#endif
#endif
