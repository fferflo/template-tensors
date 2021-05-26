namespace template_tensors {

#define ThisType CrossProduct<TVectorTypeLeft, TVectorTypeRight>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::combine<mem::memorytype_v<TVectorTypeLeft>::value, mem::memorytype_v<TVectorTypeRight>::value>(), \
                                        template_tensors::DimSeq<3> \
                              >

template <typename TVectorTypeLeft, typename TVectorTypeRight>
class CrossProduct : public SuperType
{
public:
  static_assert(is_vector_v<TVectorTypeLeft>::value && is_vector_v<TVectorTypeRight>::value, "TVectorTypeLeft and TVectorTypeRight must be vectors");

  static_assert(are_compatible_dimseqs_v<dimseq_t<TVectorTypeLeft>, dimseq_t<TVectorTypeRight>, dimseq_t<SuperType>>::value, "Incompatible dimensions");

  __host__ __device__
  CrossProduct(TVectorTypeLeft left, TVectorTypeRight right)
    : SuperType(3)
    , m_left(left)
    , m_right(right)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType,
    typename TElementType = decltype(std::declval<TThisType&&>().m_left() * std::declval<TThisType&&>().m_right())>
  __host__ __device__
  static TElementType getElement(TThisType&& self, dim_t row)
  {
    switch (row)
    {
      case 0: return self.m_left(1) * self.m_right(2) - self.m_left(2) * self.m_right(1);
      case 1: return self.m_left(2) * self.m_right(0) - self.m_left(0) * self.m_right(2);
      case 2: return self.m_left(0) * self.m_right(1) - self.m_left(1) * self.m_right(0);
      default: return 0;
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 1)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex == 0 ? 3 : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index == 0 ? 3 : 1;
  }

private:
  TVectorTypeLeft m_left;
  TVectorTypeRight m_right;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(CrossProduct<decltype(transform(m_left)), decltype(transform(m_right))>
    (transform(m_left), transform(m_right))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(CrossProduct<decltype(transform(m_left)), decltype(transform(m_right))>
    (transform(m_left), transform(m_right))
  )
};
#undef SuperType
#undef ThisType



/*!
 * \brief Returns the vector cross product of the input vectors
 * \ingroup MiscTensorOps
 *
 * @param left the left input vector
 * @param right the right input vector
 * @return the vector cross product of the input vectors
 */
template <typename TVectorTypeLeft, typename TVectorTypeRight>
__host__ __device__
auto cross(TVectorTypeLeft&& left, TVectorTypeRight&& right)
RETURN_AUTO(CrossProduct<util::store_member_t<TVectorTypeLeft&&>, util::store_member_t<TVectorTypeRight&&>>
  (std::forward<TVectorTypeLeft>(left), std::forward<TVectorTypeRight>(right))
);



#define ThisType CrossProductMatrix<TVectorType>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::memorytype_v<TVectorType>::value, \
                                        template_tensors::DimSeq<3, 3> \
                              >

template <typename TVectorType>
class CrossProductMatrix : public SuperType
{
public:
  static_assert(is_vector_v<TVectorType>::value, "TVectorType must be a vector");

  __host__ __device__
  CrossProductMatrix(TVectorType vector)
    : SuperType(3)
    , m_vector(vector)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename TElementType = typename std::decay<decltype(std::declval<TThisType&&>().m_vector())>::type>
  __host__ __device__
  static TElementType getElement(TThisType&& self, dim_t row, dim_t col)
  {
    if (row == col)
    {
      return 0;
    }
    else if (row + col == 1)
    {
      return row > col ? self.m_vector(2) : -self.m_vector(2);
    }
    else if (row + col == 2)
    {
      return row > col ? -self.m_vector(1) : self.m_vector(1);
    }
    else
    {
      ASSERT(row + col == 3, "Should not happen");
      return row > col ? self.m_vector(0) : -self.m_vector(0);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 2)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex < 2 ? 3 : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index < 2 ? 3 : 1;
  }

private:
  TVectorType m_vector;

public:
  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform)
  RETURN_AUTO(CrossProductMatrix<decltype(transform(m_vector))>
    (transform(m_vector))
  )

  HD_WARNING_DISABLE
  template <typename TTransform>
  __host__ __device__
  auto map(TTransform transform) const
  RETURN_AUTO(CrossProductMatrix<decltype(transform(m_vector))>
    (transform(m_vector))
  )
};
#undef SuperType
#undef ThisType



// matmul(cross_matrix(A), B) == cross(A, B)
template <typename TVectorType>
__host__ __device__
auto cross_matrix(TVectorType&& vector)
RETURN_AUTO(CrossProductMatrix<util::store_member_t<TVectorType&&>>
  (std::forward<TVectorType>(vector))
);

} // end of ns template_tensors
