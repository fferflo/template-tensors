namespace template_tensors {

template <typename TThisType, mem::MemoryType TMemoryType, typename TDimSeq>
class TensorBase;

class IsTensor {};

template <typename TArg>
struct is_tensor_v
{
  TMP_IF(const IsTensor&)
  TMP_RETURN_VALUE(true)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(typename std::decay<TArg>::type);
};

template <typename TArg>
struct is_matrix_v
{
  template <typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
  TMP_IF(TTensorType&&)
  TMP_RETURN_VALUE(non_trivial_dimensions_num_v<TTensorType>::value <= 2)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

template <typename TArg>
struct is_vector_v
{
  template <typename TTensorType, ENABLE_IF(template_tensors::is_tensor_v<TTensorType>::value)>
  TMP_IF(TTensorType&&)
  TMP_RETURN_VALUE(non_trivial_dimensions_num_v<TTensorType>::value <= 1)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

} // end of ns template_tensors
