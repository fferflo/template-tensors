#ifdef BOOST_NUMPY_INCLUDED

namespace template_tensors {

namespace boost {

namespace python {

namespace dispatch {

template <typename TElementTypes, typename TRanks>
struct FromNumpy
{
  struct ElementTypeMatches
  {
    ::boost::python::numpy::ndarray& array;

    template <typename TElementType>
    bool operator()()
    {
      template_tensors::boost::python::with_gil guard;
      return array.get_dtype() == ::boost::python::numpy::dtype::get_builtin<TElementType>();
    }
  };

  struct RankMatches
  {
    ::boost::python::numpy::ndarray& array;

    template <metal::int_ TRank>
    bool operator()()
    {
      template_tensors::boost::python::with_gil guard;
      return array.get_nd() == TRank;
    }
  };

  static constexpr const char* const ELEMENT_TYPE_STR = "Element type";
  static constexpr const char* const RANK_STR = "Rank";

  static auto inner_dispatcher(::boost::python::numpy::ndarray& ndarray)
  RETURN_AUTO(
    ::dispatch::all(
      ::dispatch::first_type<TElementTypes>(ElementTypeMatches{ndarray}, ELEMENT_TYPE_STR),
      ::dispatch::first_value<metal::int_, TRanks>(RankMatches{ndarray}, RANK_STR)
    )
  )

  using InnerDispatcher = decltype(inner_dispatcher(std::declval<::boost::python::numpy::ndarray&>()));

  ::boost::python::object& object;

  FromNumpy(::boost::python::object& object)
    : object(object)
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
        return prefix + "NumpyArray: " + message;
      }
      else
      {
        return prefix + "NumpyArray:\n" + inner_result.error(prefix + "  ");
      }
    }
  };

  template <typename TFunctor>
  struct Forward
  {
    ::boost::python::numpy::ndarray& ndarray;
    TFunctor functor;

    template <typename TElementType, metal::int_ TRank>
    void operator()(metal::value<TElementType>, std::integral_constant<metal::int_, TRank>)
    {
      functor(template_tensors::boost::python::fromNumpy<TElementType, TRank>(ndarray));
    }
  };

  template <typename TFunctor>
  Result operator()(TFunctor&& functor)
  {
    Result result;
    std::string class_name = template_tensors::boost::python::getClassName(object);
    if (class_name == "ndarray")
    {
      ::boost::python::numpy::ndarray& ndarray = static_cast<::boost::python::numpy::ndarray&>(object);
      result.inner_result = inner_dispatcher(ndarray)(Forward<TFunctor&&>{ndarray, util::forward<TFunctor>(functor)});
    }
    else
    {
      result.message = std::string("'") + class_name + "' is not a numpy array";
    }
    return result;
  }
};

} // end of ns dispatch

} // end of ns python

} // end of ns boost

} // end of ns template_tensors

#endif
