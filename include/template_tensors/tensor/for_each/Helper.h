namespace template_tensors {

namespace op {

#define TT_FOR_EACH_MAP_AND_COPY(...) \
template <typename TOperation, typename TTensorDest, typename... TTensorSrcs> \
__VA_ARGS__ \
static void map(TOperation&& op, TTensorDest&& dest, TTensorSrcs&&... srcs) \
{ \
  for_each(util::functor::assign_mapped<util::store_member_t<TOperation&&>>(util::forward<TOperation>(op)), util::forward<TTensorDest>(dest), util::forward<TTensorSrcs>(srcs)...); \
} \
 \
template <typename TTensorDest, typename TTensorSrc> \
__VA_ARGS__ \
static void copy(TTensorDest&& dest, TTensorSrc&& src) \
{ \
  map(math::functor::id(), util::forward<TTensorDest>(dest), util::forward<TTensorSrc>(src)); \
}

namespace detail {

template <typename TArg>
struct tensor_indexstrategy_can_convert_from_index_v
{
  template <typename TTensorType, typename TIndexStrategy = indexstrategy_t<TTensorType>>
  TMP_IF(TTensorType&&)
  TMP_RETURN_VALUE(indexstrategy_can_convert_from_index_v<TIndexStrategy>::value)

  TMP_ELSE()
  TMP_RETURN_VALUE(false)

  TMP_DEDUCE_VALUE(TArg);
};

} // end of ns detail

} // end of ns op

} // end of ns template_tensors
