#pragma once

namespace template_tensors {

template <typename TArg, typename TChar>
struct is_string_v
{
  template <typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
  TMP_IF(TTensorType&&)
  TMP_RETURN_VALUE(std::is_same<template_tensors::decay_elementtype_t<TTensorType>, TChar>::value && template_tensors::non_trivial_dimensions_num_v<TTensorType>::value <= 1)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

static_assert(is_string_v<template_tensors::AllocVectorT<char, mem::alloc::heap, template_tensors::ColMajor>, char>::value, "is_string_v not working");
static_assert(!is_string_v<double, char>::value, "is_string_v not working");

} // end of ns tensor
