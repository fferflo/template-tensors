#ifdef DLPACK_INCLUDED
#ifdef BOOST_PYTHON_INCLUDED

#include <boost/algorithm/string/join.hpp>

namespace template_tensors {

namespace boost {

namespace python {

namespace dispatch {

template <typename TElementTypes, typename TRanks, typename TMemoryTypes>
struct FromTensorflow
{
  using InnerDispatcher = template_tensors::boost::python::dispatch::FromDlPack<TElementTypes, TRanks, TMemoryTypes>;

  ::boost::python::object& object;

  FromTensorflow(::boost::python::object& object)
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
        return prefix + "Tensorflow: " + message;
      }
      else
      {
        return prefix + "Tensorflow:\n" + inner_result.error(prefix + "  ");
      }
    }
  };

  template <typename TFunctor>
  Result operator()(TFunctor&& functor)
  {
    Result result;
    std::string class_name = template_tensors::boost::python::getClassName(object);
    if (class_name == "EagerTensor")
    {
      ::boost::python::object dlpack;
      {
        template_tensors::boost::python::with_gil guard;
        ::boost::python::object tensorflow_module = ::boost::python::import("tensorflow");
        dlpack = tensorflow_module.attr("experimental").attr("dlpack").attr("to_dlpack")(object);
      }
      result.inner_result = InnerDispatcher(dlpack, "dltensor")(util::forward<TFunctor>(functor));
    }
    else
    {
      result.message = std::string("'") + class_name + "' is not a tensorflow EagerTensor";
    }
    return result;
  }
};

} // end of ns dispatch

} // end of ns python

} // end of ns boost

} // end of ns tensor

#endif
#endif
