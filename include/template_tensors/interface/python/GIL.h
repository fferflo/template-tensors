#pragma once

#ifdef PYTHON_INCLUDED

#include <Python.h>

namespace template_tensors::python {

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

} // end of ns template_tensors::python

#endif