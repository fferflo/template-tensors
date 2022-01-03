#pragma once

#ifdef BOOST_PYTHON_INCLUDED // TODO: dont check this inside interface headers, add autoinclude switch in cmake and include with check in TemplateTensors.h

#include <boost/python.hpp>
#include <boost/python/extract.hpp>

namespace template_tensors::python::boost {

namespace functor {

struct make_tuple
{
  template <typename... TArgs>
  __host__
  ::boost::python::tuple operator()(TArgs&&... args)
  {
    return ::boost::python::make_tuple((static_cast<size_t>(std::forward<TArgs>(args)))...);
  }
};

} // end of ns functor

::boost::python::object dir(::boost::python::object object)
{
  ::boost::python::handle<> handle(PyObject_Dir(object.ptr()));
  return ::boost::python::object(handle);
}

bool callable(::boost::python::object object)
{
  return 1 == PyCallable_Check(object.ptr());
}

std::string getClassName(::boost::python::object object)
{
  return ::boost::python::extract<std::string>(object.attr("__class__").attr("__name__"));
}

std::vector<std::string> getAttributes(::boost::python::object object)
{
  std::vector<std::string> result;
  for (::boost::python::stl_input_iterator<::boost::python::str> name(dir(object)), end; name != end; ++name)
  {
    if (!name->startswith("__") && !callable(object.attr(*name)))
    {
      result.push_back((std::string) ::boost::python::extract<std::string>(*name));
    }
  }
  return result;
}

} // end of ns template_tensors::python::boost

#endif
