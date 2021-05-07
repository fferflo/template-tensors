#pragma once

#ifdef BOOST_PYTHON_INCLUDED

#include <boost/python.hpp>
#include <boost/python/extract.hpp>

namespace template_tensors {

namespace boost {

namespace python {

class with_gil
{
public:
  with_gil()
  {
    state = PyGILState_Ensure();
  }

  ~with_gil()
  {
    PyGILState_Release(state);
  }

  with_gil(const with_gil&) = delete;
  with_gil& operator=(const with_gil&) = delete;

private:
  PyGILState_STATE state;
};

class without_gil
{
public:
  without_gil()
  {
    state = PyEval_SaveThread();
  }

  ~without_gil()
  {
    PyEval_RestoreThread(state);
  }

  without_gil(const without_gil&) = delete;
  without_gil& operator=(const without_gil&) = delete;

private:
  PyThreadState* state;
};

namespace functor {

struct make_tuple
{
  template <typename... TArgs>
  __host__
  ::boost::python::tuple operator()(TArgs&&... args)
  {
    template_tensors::boost::python::with_gil guard;
    return ::boost::python::make_tuple((static_cast<size_t>(util::forward<TArgs>(args)))...);
  }
};

} // end of ns functor

::boost::python::object dir(::boost::python::object object)
{
  template_tensors::boost::python::with_gil guard;
  ::boost::python::handle<> handle(PyObject_Dir(object.ptr()));
  return ::boost::python::object(handle);
}

bool callable(::boost::python::object object)
{
  template_tensors::boost::python::with_gil guard;
  return 1 == PyCallable_Check(object.ptr());
}

std::string getClassName(::boost::python::object object)
{
  template_tensors::boost::python::with_gil guard;
  return ::boost::python::extract<std::string>(object.attr("__class__").attr("__name__"));
}

std::vector<std::string> getAttributes(::boost::python::object object)
{
  template_tensors::boost::python::with_gil guard;
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

} // end of ns python

} // end of ns boost

} // end of ns tensor

#endif
