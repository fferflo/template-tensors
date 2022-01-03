#if defined(DLPACK_INCLUDED) && defined(BOOST_PYTHON_INCLUDED)

#include <boost/python.hpp>

namespace template_tensors::python::boost {

namespace detail {

inline void deleteDlPackCapsuleObject(PyObject* capsule)
{
  DLManagedTensor* dl = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule)));
  ASSERT_(dl != nullptr, "Capsule object is null"); // TODO: assertion levels
  if (dl->deleter != nullptr)
  {
    dl->deleter(dl);
  }
  PyCapsule_SetDestructor(capsule, nullptr);
}

} // end of ns detail

inline ::boost::python::object fromDlPack(template_tensors::SafeDLManagedTensor&& dl, const char* name)
{
  void* dl_ptr = reinterpret_cast<void*>(dl.use());
  ASSERT_(dl_ptr != nullptr, "Cannot create capsule with nullptr ptr"); // TODO: assertion levels
  PyObject* capsule = ::PyCapsule_New(dl_ptr, name, (PyCapsule_Destructor) &detail::deleteDlPackCapsuleObject);
  ::boost::python::handle<> capsule_handle{capsule};
  ::boost::python::object capsule_object{capsule_handle};
  return capsule_object;
}

inline template_tensors::SafeDLManagedTensor toDlPack(::boost::python::object capsule, const char* name)
{
  DLManagedTensor* dl = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), name));
  ASSERT_(dl != nullptr, "Capsule object is null"); // TODO: assertion levels
  PyCapsule_SetName(capsule.ptr(), "tensor_pycapsule_used");
  PyCapsule_SetDestructor(capsule.ptr(), nullptr);
  return template_tensors::SafeDLManagedTensor(dl);
}

} // end of ns template_tensors::python::boost

#endif
