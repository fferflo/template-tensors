namespace template_tensors {
/* TODO: refactor this
namespace sparse {

template <typename TElementType, size_t TRank>
class CoordinateElement
{
public:
  __host__ __device__
  CoordinateElement(TElementType element, VectorXs<TRank> coordinates)
    : m_element(element)
    , m_coordinates(coordinates)
  {
  }

  __host__ __device__
  TElementType& getElement()
  {
    return m_element;
  }

  __host__ __device__
  const TElementType& getElement() const
  {
    return m_element;
  }

  __host__ __device__
  const VectorXs<TRank>& getCoordinates() const
  {
    return m_coordinates;
  }

private:
  TElementType m_element;
  VectorXs<TRank> m_coordinates;
};

#define ThisType IterableEx<TElementType, TIterable, TRank, TDimSeq>
#define SuperType TensorBase< \
                                        ThisType, \
                                        mem::LOCAL, \
                                        TDimSeq \
                              >
template <typename TElementType, template <typename> class TIterable, size_t TRank, typename TDimSeq>
class IterableEx : public SuperType, public StoreDimensions<TDimSeq>
{
public:
  HD_WARNING_DISABLE
  template <typename... TDimArgTypes>
  __host__ __device__
  IterableEx(TElementType default_element, TDimArgTypes&&... dim_args)
    : SuperType(util::forward<TDimArgTypes>(dim_args)...)
    , StoreDimensions<TDimSeq>(util::forward<TDimArgTypes>(dim_args)...)
    , m_iterable()
    , m_default(default_element)
  {
  }

  HD_WARNING_DISABLE
  template <typename... TDimArgTypes>
  __host__ __device__
  IterableEx(const TIterable<CoordinateElement<TElementType, TRank>>& iterable, TElementType default_element, TDimArgTypes&&... dim_args)
    : SuperType(util::forward<TDimArgTypes>(dim_args)...)
    , m_iterable(iterable)
    , m_default(default_element)
  {
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__ __device__
  static TElementType getElement(TThisType&& self, TCoordArgTypes&&... coords)
  {
    for (CoordinateElement<TElementType, TRank>& el : self.m_iterable)
    {
      if (areSameCoordinates2(el.getCoordinates(), util::forward<TCoordArgTypes>(coords)...))
      {
        return el.getElement();
      }
    }
    return self.m_default;
  }
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  __host__ __device__
  TIterable<CoordinateElement<TElementType, TRank>>& getIterable()
  {
    return m_iterable;
  }

  __host__ __device__
  const TIterable<CoordinateElement<TElementType, TRank>>& getIterable() const
  {
    return m_iterable;
  }

  HD_WARNING_DISABLE
  template <typename... TCoordArgTypes>
  __host__ __device__
  void push_back(TElementType element, TCoordArgTypes&&... coords)
  {
    m_iterable.push_back(CoordinateElement<TElementType, TRank>(element, toCoordVector<TRank>(util::forward<TCoordArgTypes>(coords)...)));
  }

private:
  TIterable<CoordinateElement<TElementType, TRank>> m_iterable;
  TElementType m_default;

public:
  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform)
  {
    return ThisType(*this);
  }
  // TODO: this transform for iterable types, so that thrust::device_vector is transformed by referring to memory instead of copying
  template <typename TTransform>
  __host__ __device__
  ThisType map(TTransform transform) const
  {
    return ThisType(*this);
  }
};
#undef SuperType
#undef ThisType

template <typename TElementType, template <typename> class TIterable, size_t... TDims>
using Iterable = IterableEx<TElementType, TIterable, sizeof...(TDims), DimSeq<TDims...>>;

} // end of ns sparse
*/
} // end of ns tensor
