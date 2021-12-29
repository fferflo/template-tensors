#if defined(DLPACK_INCLUDED) && defined(BOOST_PYTHON_INCLUDED)

namespace template_tensors::python::boost::dispatch {

template <typename TElementTypes, typename TRanks, typename TMemoryTypes>
struct FromTensorflow
{
  using InnerDispatcher = template_tensors::python::boost::dispatch::FromDlPack<TElementTypes, TRanks, TMemoryTypes>;

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
    std::string class_name = template_tensors::python::boost::getClassName(object);
    if (class_name == "EagerTensor")
    {
      ::boost::python::object dlpack;
      {
        template_tensors::python::with_gil guard;
        ::boost::python::object tensorflow_module = ::boost::python::import("tensorflow");
        dlpack = tensorflow_module.attr("experimental").attr("dlpack").attr("to_dlpack")(object);
      }
      result.inner_result = InnerDispatcher(dlpack, "dltensor")(std::forward<TFunctor>(functor));
    }
    else
    {
      result.message = std::string("'") + class_name + "' is not a tensorflow EagerTensor";
    }
    return result;
  }
};

} // end of ns template_tensors::python::boost::dispatch

#endif
