#pragma once

namespace field {

template <metal::int_ TRank, typename TSupplier>
class FromSupplierField
{
public:
  static const metal::int_ RANK = TRank;

private:
  TSupplier m_supplier;

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(
    util::forward<TThisType>(self).m_supplier(util::forward<TCoordVector>(coords))
  )

public:
  __host__ __device__
  FromSupplierField(TSupplier supplier)
    : m_supplier(supplier)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get)
};

template <metal::int_ TRank, typename TSupplier>
__host__ __device__
auto fromSupplier(TSupplier&& supplier)
RETURN_AUTO(FromSupplierField<TRank, util::store_member_t<TSupplier&&>>(util::forward<TSupplier>(supplier)))

} // end of ns field
