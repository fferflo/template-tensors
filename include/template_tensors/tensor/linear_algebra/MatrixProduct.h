namespace template_tensors {

namespace detail {

template <metal::int_ TIndex>
struct MatrixProductDimHelper
{
  template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
  __host__ __device__
  static dim_t get(TMatrixTypeLeft&& left, TMatrixTypeRight&& right)
  {
    return 1;
  }
};

template <>
struct MatrixProductDimHelper<0>
{
  template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
  __host__ __device__
  static dim_t get(TMatrixTypeLeft&& left, TMatrixTypeRight&& right)
  {
    return left.template dim<0>();
  }
};

template <>
struct MatrixProductDimHelper<1>
{
  template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
  __host__ __device__
  static dim_t get(TMatrixTypeLeft&& left, TMatrixTypeRight&& right)
  {
    return right.template dim<1>();
  }
};

template <typename TVectorElementTypeLeft, typename TVectorElementTypeRight>
struct MatrixProductElementType2
{
  using type = decltype(std::declval<TVectorElementTypeLeft>() * std::declval<TVectorElementTypeRight>());
};

template <typename TVectorElementTypeLeft>
struct MatrixProductElementType2<TVectorElementTypeLeft, void>
{
  using type = void;
};

template <typename TVectorElementTypeRight>
struct MatrixProductElementType2<void, TVectorElementTypeRight>
{
  using type = void;
};

template <>
struct MatrixProductElementType2<void, void>
{
  using type = void;
};

template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
struct MatrixProductElementType
{
  using type = typename MatrixProductElementType2<decltype(std::declval<TMatrixTypeLeft>()()), decltype(std::declval<TMatrixTypeRight>()())>::type;
};

} // end of ns detail

#define ThisType MatrixProduct<TMatrixTypeLeft, TMatrixTypeRight>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::combine<mem::memorytype_v<TMatrixTypeLeft>::value, mem::memorytype_v<TMatrixTypeRight>::value>(), \
                                        template_tensors::DimSeq<rows_v<TMatrixTypeLeft>::value, cols_v<TMatrixTypeRight>::value> \
                              >
template <typename TMatrixTypeLeft, typename TMatrixTypeRight>
class MatrixProduct : public SuperType
{
public:
  static_assert(is_matrix_v<TMatrixTypeLeft>::value && is_matrix_v<TMatrixTypeRight>::value, "TMatrixTypeLeft and TMatrixTypeRight must be matrices");
  static_assert(are_compatible_dimseqs_v<TMatrixTypeLeft, template_tensors::DimSeq<DYN, DYN>>::value, "Incompatible dimensions");
  static_assert(are_compatible_dimseqs_v<TMatrixTypeRight, template_tensors::DimSeq<DYN, DYN>>::value, "Incompatible dimensions");
  static_assert(are_compatible_dimseqs_v<
    template_tensors::DimSeq<cols_v<TMatrixTypeLeft>::value>,
    template_tensors::DimSeq<rows_v<TMatrixTypeRight>::value>
  >::value, "Incompatible dimensions");

  __host__ __device__
  MatrixProduct(TMatrixTypeLeft left, TMatrixTypeRight right)
    : SuperType(left.template dim<0>(), right.template dim<1>())
    , m_left(left)
    , m_right(right)
  {
    ASSERT(left.template dim<1>() == right.template dim<0>(), "Incompatible dimensions");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType,
    typename TElementType = decltype(std::declval<TThisType&&>().m_left() * std::declval<TThisType&&>().m_right())>
  __host__ __device__
  static TElementType getElement(TThisType&& self, dim_t row, dim_t col)
  {
    TElementType sum = 0;
    for (dim_t k = 0; k < self.m_left.template dim<1>(); k++)
    {
      sum += self.m_left(row, k) * self.m_right(k, col);
    }
    return sum;
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 2)
  // TODO: replace this with dotproduct between template_tensors::row and template_tensors::col call?
  // TODO: all self.member should be rvalue if self is rvalue?

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return detail::MatrixProductDimHelper<TIndex>::get(m_left, m_right);
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    switch (index)
    {
      case 0: return m_left.template dim<0>();
      case 1: return m_right.template dim<1>();
      default: return 1;
    }
  }

private:
  TMatrixTypeLeft m_left;
  TMatrixTypeRight m_right;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(MatrixProduct<decltype(transform(m_left)), decltype(transform(m_right))>
    (transform(m_left), transform(m_right))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(MatrixProduct<decltype(transform(m_left)), decltype(transform(m_right))>
    (transform(m_left), transform(m_right))
  )
};
#undef SuperType
#undef ThisType



template <typename TMatrixType>
__host__ __device__
auto matmul(TMatrixType&& matrix)
RETURN_AUTO(util::forward_lvalue<TMatrixType>(matrix))

template <typename TMatrixType1, typename TMatrixType2, typename... TMatrixTypeRest>
__host__ __device__
auto matmul(TMatrixType1&& m1, TMatrixType2&& m2, TMatrixTypeRest&&... rest)
RETURN_AUTO(template_tensors::matmul(
  MatrixProduct<util::store_member_t<TMatrixType1&&>, util::store_member_t<TMatrixType2&&>>(util::forward<TMatrixType1>(m1), util::forward<TMatrixType2>(m2)),
  util::forward<TMatrixTypeRest>(rest)...
))

template <typename TMatrixType>
__host__ __device__
auto matmul_eval(TMatrixType&& matrix)
RETURN_AUTO(util::forward_lvalue<TMatrixType>(matrix))

template <typename TMatrixType1, typename TMatrixType2, typename... TMatrixTypeRest>
__host__ __device__
auto matmul_eval(TMatrixType1&& m1, TMatrixType2&& m2, TMatrixTypeRest&&... rest)
RETURN_AUTO(matmul_eval(
  template_tensors::eval(template_tensors::matmul(util::forward<TMatrixType1>(m1), util::forward<TMatrixType2>(m2))),
  util::forward<TMatrixTypeRest>(rest)...
))

} // end of ns tensor
