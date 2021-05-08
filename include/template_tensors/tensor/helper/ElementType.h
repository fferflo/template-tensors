namespace template_tensors {

namespace detail {

template <typename... TTensorTypes>
struct CombineTensorMemberElementTypesHelper
{
  static_assert(math::land(!std::is_rvalue_reference<decltype(std::declval<TTensorTypes>()())>::value...), "Cannot be rvalue references");
  // TODO: why not rvalue references?
  using common_nocv = typename std::common_type<typename std::remove_reference<decltype(std::declval<TTensorTypes>()())>::type...>::type;
  using common_noc = typename std::conditional<math::lor(std::is_volatile<typename std::remove_reference<decltype(std::declval<TTensorTypes>()())>::type>::value...),
      volatile common_nocv,
      common_nocv
    >::type;
  using common = typename std::conditional<math::lor(std::is_const<typename std::remove_reference<decltype(std::declval<TTensorTypes>()())>::type>::value...),
      const common_noc,
      common_noc
    >::type;

  using type = typename std::conditional<math::land(std::is_reference<decltype(std::declval<TTensorTypes>()())>::value...),
    common&,
    common
  >::type;
};

} // end of ns detail

template <typename TTensorRefType>
using decay_elementtype_t = typename std::decay<decltype(std::declval<TTensorRefType>()())>::type;

template <typename... TTensorTypes>
using common_elementtype_t = typename detail::CombineTensorMemberElementTypesHelper<TTensorTypes...>::type;

} // end of ns template_tensors
