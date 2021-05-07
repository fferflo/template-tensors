#pragma once

#if CPPFLOW_INCLUDED

#include <cppflow/cppflow.h>
#include <tensorflow/c/eager/dlpack.h>

namespace template_tensors {

template <typename TElementType, metal::int_ TRank, mem::MemoryType TMemoryType>
__host__
FromDlPack<TElementType, TRank, TMemoryType> fromCppFlow(cppflow::tensor cppflow)
{
  void* dl_ptr = tensorflow::TFE_HandleToDLPack(cppflow.tfe_handle.get(), cppflow::context::get_status());
  cppflow::status_check(cppflow::context::get_status());
  SafeDLManagedTensor dlpack = SafeDLManagedTensor(reinterpret_cast<DLManagedTensor*>(dl_ptr));
  return fromDlPack<TElementType, TRank, TMemoryType>(util::move(dlpack));
}

template <typename TTensorType>
__host__
cppflow::tensor toCppFlow(TTensorType&& tensor)
{
  SafeDLManagedTensor dlpack = template_tensors::toDlPack(util::forward<TTensorType>(tensor));
  TFE_TensorHandle* tfe_handle = tensorflow::TFE_HandleFromDLPack(dlpack.use(), cppflow::context::get_status(), cppflow::context::get_context());
  cppflow::status_check(cppflow::context::get_status());
  return cppflow::tensor(tfe_handle);
}

} // end of ns tensor

#endif
