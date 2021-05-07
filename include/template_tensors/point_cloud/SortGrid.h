#include <metal.hpp>

namespace point_cloud {

namespace detail {

template <typename TObjectVector>
struct SortGridCellEntry
{
  using Object = typename std::remove_reference<decltype(std::declval<TObjectVector>()())>::type;

  Object* first;
  uint32_t num;

  __host__ __device__
  SortGridCellEntry()
    : first(nullptr)
    , num(0)
  {
  }

  __host__ __device__
  SortGridCellEntry(Object* first, size_t num)
    : first(first)
    , num(num)
  {
  }

  __host__ __device__
  Object* begin()
  {
    return first;
  }

  __host__ __device__
  const Object* begin() const
  {
    return first;
  }

  __host__ __device__
  Object* end()
  {
    return first + num;
  }

  __host__ __device__
  const Object* end() const
  {
    return first + num;
  }
};

} // end of ns detail

#define ThisType SortGrid<TObjectVector, TRank, TAllocator, TIndexStrategy, TAtomicOpsIn, TIsOnHost>
#define SuperType template_tensors::TensorBase< \
                                        ThisType, \
                                        TAllocator::MEMORY_TYPE, \
                                        template_tensors::dyn_dimseq_t<TRank> \
                              >
// TODO: make this multithreaded for cpu: modify ::atomic::op::Void and for_each calls
template <typename TObjectVector, metal::int_ TRank, typename TAllocator, typename TIndexStrategy = template_tensors::RowMajor,
  typename TAtomicOpsIn = void, bool TIsOnHost = TT_IS_ON_HOST>
class SortGrid : public SuperType
{
public:
  static const mem::MemoryType MEMORY_TYPE = TAllocator::MEMORY_TYPE;

private:
  using SortGridCellEntry = detail::SortGridCellEntry<TObjectVector>;

public:
  using Object = typename SortGridCellEntry::Object;

private:
  using AtomicOps =
#ifdef __CUDACC__
  typename std::conditional<std::is_same<TAtomicOpsIn, void>::value,
    typename std::conditional<mem::isOnDevice<MEMORY_TYPE, TIsOnHost>(), cuda::AtomicOps<>, ::atomic::op::Void>::type,
    TAtomicOpsIn
  >::type;
#else
  ::atomic::op::Void;
#endif

#ifdef __CUDACC__
  TObjectVector m_ordered_objects;
  template_tensors::FromThrustVector<thrust::vector_for<MEMORY_TYPE, size_t>> m_ordered_cell_ids;
#else
  TObjectVector m_ordered_objects;
  template_tensors::FromStdVector<std::vector_for<MEMORY_TYPE, size_t>> m_ordered_cell_ids;
#endif
  template_tensors::AllocTensorT<SortGridCellEntry, TAllocator, TIndexStrategy, TRank> m_grid;
  TIndexStrategy m_index_strategy;
  AtomicOps m_atomic_ops;

public:
  template <typename TObjectVectorArg>
  __host__
  SortGrid(TObjectVectorArg&& object_vector_arg, template_tensors::VectorXs<TRank> resolution, TIndexStrategy index_strategy = TIndexStrategy(), AtomicOps atomic_ops = AtomicOps())
    : SuperType(resolution)
    , m_ordered_objects(util::forward<TObjectVectorArg>(object_vector_arg))
    , m_ordered_cell_ids(m_ordered_objects.rows())
    , m_grid(index_strategy, resolution)
    , m_index_strategy(index_strategy)
    , m_atomic_ops(atomic_ops)
  {
  }

  __host__
  SortGrid(template_tensors::VectorXs<TRank> resolution, TIndexStrategy index_strategy = TIndexStrategy(), AtomicOps atomic_ops = AtomicOps())
    : SortGrid(resolution, 0, index_strategy, atomic_ops)
  {
  }

  __host__
  SortGrid(const SortGrid<TObjectVector, TRank, TAllocator, TIndexStrategy, TAtomicOpsIn, TIsOnHost>& other)
    : SuperType(static_cast<const SuperType&>(other))
    , m_ordered_objects(other.m_ordered_objects)
    , m_ordered_cell_ids(other.m_ordered_cell_ids)
    , m_grid(other.m_grid)
    , m_index_strategy(other.m_index_strategy)
    , m_atomic_ops(other.m_atomic_ops)
  {
  }

  __host__
  SortGrid(SortGrid<TObjectVector, TRank, TAllocator, TIndexStrategy, TAtomicOpsIn, TIsOnHost>&& other)
    : SuperType(static_cast<SuperType&&>(other))
    , m_ordered_objects(util::move(other.m_ordered_objects))
    , m_ordered_cell_ids(util::move(other.m_ordered_cell_ids))
    , m_grid(util::move(other.m_grid))
    , m_index_strategy(util::move(other.m_index_strategy))
    , m_atomic_ops(util::move(other.m_atomic_ops))
  {
  }

  __host__
  ~SortGrid()
  {
  }

  template <typename TGetCoordinates>
  __host__
  void update(TGetCoordinates&& get_coordinates)
  {
    m_ordered_cell_ids.getVector().resize(m_ordered_objects.rows());

    // Create cell and object vectors
    TIndexStrategy index_strategy = m_index_strategy;
    template_tensors::VectorXs<TRank> dims = m_grid.template dims<TRank>();
    auto get_coordinates_to_functor = util::wrap(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(util::forward<TGetCoordinates>(get_coordinates)));
    template_tensors::for_each([get_coordinates_to_functor, index_strategy, dims]
    __host__ __device__(size_t& cell_id, Object object_in){
        cell_id = index_strategy.toIndex(dims, get_coordinates_to_functor()(object_in));
      }, m_ordered_cell_ids, m_ordered_objects);

    // Sort cell and object vectors
#ifdef __CUDACC__
    util::constexpr_if<mem::isOnDevice<MEMORY_TYPE, TIsOnHost>()>(
      [&](){
        thrust::sort_by_key(thrust::device, m_ordered_cell_ids.begin(), m_ordered_cell_ids.end(), m_ordered_objects.begin());
      },
      [&](){
        thrust::sort_by_key(thrust::host, m_ordered_cell_ids.begin(), m_ordered_cell_ids.end(), m_ordered_objects.begin());
      }
    );
#else
    std::sort_by_key(m_ordered_cell_ids.begin(), m_ordered_cell_ids.end(), m_ordered_objects.begin());
#endif


    // Create pointer grid
    template_tensors::fill(m_grid, SortGridCellEntry());
    auto ordered_cell_ids_to_functor = util::wrap(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(m_ordered_cell_ids));
    auto grid_to_functor = util::wrap(mem::toFunctor<MEMORY_TYPE, TIsOnHost>(m_grid));
    auto atomic_ops = m_atomic_ops;
    template_tensors::for_each<1>([index_strategy, dims, ordered_cell_ids_to_functor, grid_to_functor, atomic_ops]
    __host__ __device__(template_tensors::Vector1s index, size_t cell_id, Object& object) mutable {
      template_tensors::VectorXs<TRank> cell = index_strategy.fromIndex(cell_id, dims);
      SortGridCellEntry& cell_entry = grid_to_functor()(cell);
      atomic_ops.add(cell_entry.num, 1);
      if (index() == 0 || ordered_cell_ids_to_functor()(index() - 1) != cell_id)
      {// TODO: this can also be done using atomicMin?
        cell_entry.first = &object;
      }
    }, m_ordered_cell_ids, m_ordered_objects);

    INSTANTIATE_HOST(INSTANTIATE_ARG(SortGrid<TObjectVector, TRank, TAllocator, TIndexStrategy, TAtomicOpsIn, true>).template update,
      INSTANTIATE_ARG(TGetCoordinates&&));
    INSTANTIATE_HOST(INSTANTIATE_ARG(SortGrid<TObjectVector, TRank, TAllocator, TIndexStrategy, TAtomicOpsIn, MEMORY_TYPE == mem::HOST>).template update,
      INSTANTIATE_ARG(TGetCoordinates&&));
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType, typename... TCoordArgTypes>
  __host__
  static auto getElement(TThisType&& self, TCoordArgTypes&&... coords)
  RETURN_AUTO(
    self.m_grid(util::forward<TCoordArgTypes>(coords)...)
  )
  TT_ARRAY_SUBCLASS_FORWARD_ELEMENT_ACCESS(getElement)

  template <metal::int_ TIndex>
  __host__
  template_tensors::dim_t getDynDim() const
  {
    return m_grid.template getDynDim<TIndex>();
  }

  __host__
  template_tensors::dim_t getDynDim(size_t index) const
  {
    return m_grid.getDynDim(index);
  }

#ifdef __CUDACC__
  template <typename TGrid>
  __host__
  static auto toKernel(TGrid&& grid)
  RETURN_AUTO(mem::toKernel(grid.m_grid))
#endif
};
#undef SuperType
#undef ThisType

} // end of ns point_cloud
