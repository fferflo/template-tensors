namespace template_tensors {

#define ThisType BasicRotationMatrix<TElementType, TAxis1, TAxis2, TRowsCols>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<TRowsCols, TRowsCols> \
                              >

template <typename TElementType, metal::int_ TAxis1, metal::int_ TAxis2, metal::int_ TRowsCols>
class BasicRotationMatrix : public SuperType
{
public:
  static_assert(TAxis1 < TAxis2, "Axis 1 must come before axis 2");
  static_assert(TAxis2 < TRowsCols, "Axis out of range");

  __host__ __device__
  BasicRotationMatrix(TElementType angle)
    : SuperType(TRowsCols, TRowsCols)
    , m_cos(math::cos(angle))
    , m_sin(math::sin(angle))
  {
  }

  __host__ __device__
  BasicRotationMatrix(TElementType angle, dim_t rows_cols)
    : SuperType(rows_cols, rows_cols)
    , m_cos(math::cos(angle))
    , m_sin(math::sin(angle))
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static TElementType getElement(TThisType&& self, dim_t row, dim_t col)
  {
    if (row == col)
    {
      return (row == TAxis1 || row == TAxis2) ? self.m_cos : static_cast<TElementType>(1);
    }
    else if (row == TAxis1 && col == TAxis2)
    {
      return ((TAxis2 - TAxis1) % 2 == 0) ? self.m_sin : -self.m_sin;
    }
    else if (row == TAxis2 && col == TAxis1)
    {
      return ((TAxis2 - TAxis1) % 2 == 1) ? self.m_sin : -self.m_sin;
    }
    else
    {
      return static_cast<TElementType>(0);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 2)

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

private:
  TElementType m_cos;
  TElementType m_sin;
};
#undef SuperType
#undef ThisType

#define ThisType BasicRotationMatrix<TElementType, TAxis1, TAxis2, DYN>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        template_tensors::DimSeq<DYN, DYN> \
                              >

template <typename TElementType, metal::int_ TAxis1, metal::int_ TAxis2>
class BasicRotationMatrix<TElementType, TAxis1, TAxis2, DYN> : public SuperType
{
public:
  static_assert(TAxis1 < TAxis2, "Axis 1 must come before axis 2");

  __host__ __device__
  BasicRotationMatrix(TElementType angle, dim_t rows_cols)
    : SuperType(rows_cols, rows_cols)
    , m_cos(math::cos(angle))
    , m_sin(math::sin(angle))
    , m_rows_cols(rows_cols)
  {
    ASSERT(TAxis2 < rows_cols, "Axis out of range");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType>
  __host__ __device__
  static TElementType getElement(TThisType&& self, dim_t row, dim_t col)
  {
    if (row == col)
    {
      return (row == TAxis1 || row == TAxis2) ? self.m_cos : static_cast<TElementType>(1);
    }
    else if (row == TAxis1 && col == TAxis2)
    {
      return ((TAxis2 - TAxis1) % 2 == 0) ? self.m_sin : -self.m_sin;
    }
    else if (row == TAxis2 && col == TAxis1)
    {
      return ((TAxis2 - TAxis1) % 2 == 1) ? self.m_sin : -self.m_sin;
    }
    else
    {
      return static_cast<TElementType>(0);
    }
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS_DIM_T_N(getElement, 2)

  template <metal::int_ TIndex>
  __host__ __device__
  dim_t getDynDim() const
  {
    return TIndex < 2 ? m_rows_cols : 1;
  }

  __host__ __device__
  dim_t getDynDim(size_t index) const
  {
    return index < 2 ? m_rows_cols : 1;
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }

  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }

private:
  TElementType m_cos;
  TElementType m_sin;
  dim_t m_rows_cols;
};
#undef SuperType
#undef ThisType



template <typename TElementType, metal::int_ TAxis>
using BasicRotationMatrix3 = BasicRotationMatrix<TElementType, TAxis == 0 ? 1 : 0, TAxis == 2 ? 1 : 2, 3>;
template <typename TElementType>
using BasicRotationMatrix2 = BasicRotationMatrix<TElementType, 0, 1, 2>;



namespace detail {

template <typename TElementType, metal::int_... TAxes>
struct EulerRotationHelper3;

template <typename TElementType, metal::int_ TAxis>
struct EulerRotationHelper3<TElementType, TAxis>
{
  __host__ __device__
  static auto make(TElementType angle)
  RETURN_AUTO(BasicRotationMatrix3<TElementType, TAxis>(angle))
};

template <typename TElementType, metal::int_ TAxis, metal::int_ TAxisNext, metal::int_... TAxes>
struct EulerRotationHelper3<TElementType, TAxis, TAxisNext, TAxes...>
{
  template <typename... TArgs>
  __host__ __device__
  static auto make(TElementType angle, TElementType next_angle, TArgs&&... args)
  RETURN_AUTO(matmul(eval(EulerRotationHelper3<TElementType, TAxisNext, TAxes...>::make(next_angle, util::forward<TArgs>(args)...)),
                     BasicRotationMatrix3<TElementType, TAxis>(angle)))
};

} // end of ns detail

/*!
 * \brief Returns the 3d euler rotation matrix of the given angles around the given axes.
 *
 * \ingroup SpecialTensorConstants
 *
 * Angles and axes are given in sequential order, where the rotations are done in the order that they are given.
 * Example: <CODE>euler_rotation_3d<float, 2, 0>(0.1, 0.2)</CODE> will result in a matrix that first rotates 0.1 radians
 * around the third axis (index 2) and then 0.2 radians around the first axis (index 0).
 *
 * @param angles... the angles of rotation, in radians
 * @tparam TElementType the desired element type of the resulting matrix
 * @tparam TAxes... the axes around which will be rotated
 * @return the 3d euler rotation matrix
 */
template <typename TElementType, metal::int_... TAxes, typename... TArgs>
__host__ __device__
auto euler_rotation_3d(TArgs&&... angles)
RETURN_AUTO(eval(detail::EulerRotationHelper3<TElementType, TAxes...>::template make(util::forward<TArgs>(angles)...)))

} // end of ns template_tensors
