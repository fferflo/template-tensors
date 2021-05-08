namespace template_tensors {

template <typename TThisType>
class Maskable
{
private:
  template <typename TThisType2, typename TMask>
  __host__ __device__
  static auto mask_with(TThisType2&& self, TMask&& mask)
  RETURN_AUTO(template_tensors::mask(static_cast<util::copy_qualifiers_t<TThisType, TThisType2&&>>(self), util::forward<TMask>(mask)))

public:
  FORWARD_ALL_QUALIFIERS(operator[], mask_with)
};

} // end of ns template_tensors
