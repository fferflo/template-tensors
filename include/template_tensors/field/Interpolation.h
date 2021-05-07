#pragma once

namespace interpolate {

namespace detail {

template <metal::int_ I, metal::int_ N>
struct SeparableInterpolatorHelper
{
  template <typename TMonoInterpolator, typename TScalar>
  struct Functor
  {
    TMonoInterpolator mono_interpolator;
    TScalar t;

    template <typename TThisType, typename TValuesVector>
    __host__ __device__
    static auto get(TThisType&& self, TValuesVector&& values)
    RETURN_AUTO(self.mono_interpolator(values, self.t))
    FORWARD_ALL_QUALIFIERS(operator(), get)
  };

  template <typename TMonoInterpolator, typename TTensorType, typename TWeightsVector>
  __host__ __device__
  static auto get(TMonoInterpolator&& mono_interpolator, TTensorType&& values, TWeightsVector&& t)
  RETURN_AUTO(
    SeparableInterpolatorHelper<I + 1, N>::get(
      util::forward<TMonoInterpolator>(mono_interpolator),
      template_tensors::elwise(Functor<util::store_member_t<TMonoInterpolator&&>, template_tensors::decay_elementtype_t<TWeightsVector&&>>{util::forward<TMonoInterpolator>(mono_interpolator), t(N - 1 - I)}, template_tensors::partial<N - 1 - I>(util::forward<TTensorType>(values))),
      util::forward<TWeightsVector>(t)
    )
  )
};

template <metal::int_ I>
struct SeparableInterpolatorHelper<I, I>
{
  template <typename TMonoInterpolator, typename TTensorType, typename TWeightsVector>
  __host__ __device__
  static auto get(TMonoInterpolator&& mono_interpolator, TTensorType&& values, TWeightsVector&& t)
  RETURN_AUTO(values())
};

} // end of ns detail

template <typename TMonoInterpolator>
class Separable
{
private:
  TMonoInterpolator m_mono_interpolator;

public:
  static const metal::int_ ARG_NUM = std::decay<TMonoInterpolator>::type::ARG_NUM;

  __host__ __device__
  Separable(TMonoInterpolator mono_interpolator = TMonoInterpolator())
    : m_mono_interpolator(mono_interpolator)
  {
  }

  template <typename TThisType, typename TTensorType, typename TWeightsVector>
  __host__ __device__
  static auto get(TThisType&& self, TTensorType&& values, TWeightsVector&& t)
  RETURN_AUTO(detail::SeparableInterpolatorHelper<0, template_tensors::rows_v<TWeightsVector>::value>::get(self.m_mono_interpolator, util::forward<TTensorType>(values), util::forward<TWeightsVector>(t)))
  FORWARD_ALL_QUALIFIERS(operator(), get)
};

struct Linear
{
  static const metal::int_ ARG_NUM = 2;

  template <typename TValuesVector, typename TScalar>
  __host__ __device__
  auto operator()(TValuesVector&& values, TScalar t) const volatile
  RETURN_AUTO((t == 0) ? util::forward<TValuesVector>(values)(0) : (util::forward<TValuesVector>(values)(0) * (1 - t) + util::forward<TValuesVector>(values)(1) * t))
};

struct Nearest
{
  static const metal::int_ ARG_NUM = 2;

  template <typename TValuesVector, typename TScalar>
  __host__ __device__
  auto operator()(TValuesVector&& values, TScalar t) const volatile
  RETURN_AUTO((t < 0.5) ? util::forward<TValuesVector>(values)(0) : util::forward<TValuesVector>(values)(1))
};

} // end of ns interpolate

namespace field {

template <typename TDiscreteField, typename TInterpolator, metal::int_ TRank = std::decay<TDiscreteField>::type::RANK>
class InterpolatedField
{
public:
  static const metal::int_ RANK = TRank;

private:
  TDiscreteField m_discrete_field;
  TInterpolator m_interpolator;

  static const metal::int_ ARG_NUM = std::decay<TInterpolator>::type::ARG_NUM;

  struct ShiftCoordinates
  {
    template_tensors::VectorXi<RANK> grid_coords;

    template <typename TThisType, typename TArg>
    __host__ __device__
    static auto get(TThisType&& self, TArg&& arg)
    RETURN_AUTO(util::forward<TArg>(arg) + util::forward<TThisType>(self).grid_coords)
    FORWARD_ALL_QUALIFIERS(operator(), get)
  };

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get2(TThisType&& self, TCoordVector&& coords, template_tensors::VectorXi<RANK> grid_coords)
  RETURN_AUTO(
    util::forward<TThisType>(self).m_interpolator(
      template_tensors::fromSupplier<template_tensors::repeat_dimseq_t<ARG_NUM, TRank>, TRank>(
        field::transform(util::forward<TThisType>(self).m_discrete_field, ShiftCoordinates{grid_coords})
      ),
      coords - grid_coords
    )
  )

  template <typename TThisType, typename TCoordVector>
  __host__ __device__
  static auto get1(TThisType&& self, TCoordVector&& coords)
  RETURN_AUTO(get2(util::forward<TThisType>(self), util::forward<TCoordVector>(coords), template_tensors::static_cast_to<int32_t>(coords + (-((ARG_NUM - 1) / 2) + (ARG_NUM % 2) * 0.5))))

public:
  __host__ __device__
  InterpolatedField(TDiscreteField discrete_field, TInterpolator interpolator)
    : m_discrete_field(discrete_field)
    , m_interpolator(interpolator)
  {
  }

  FORWARD_ALL_QUALIFIERS(operator(), get1)
};

namespace detail {

template <metal::int_ TInRank, typename TDiscreteField>
struct InterpolateRank
{
  static const metal::int_ value = TInRank;
};

template <typename TDiscreteField>
struct InterpolateRank<template_tensors::DYN, TDiscreteField>
{
  static const metal::int_ value = std::decay<TDiscreteField>::type::RANK;
};

} // end of ns detail

template <typename TDiscreteField, typename TInterpolator>
__host__ __device__
auto interpolate(TDiscreteField&& discrete_field, TInterpolator&& interpolator)
RETURN_AUTO(InterpolatedField<util::store_member_t<TDiscreteField&&>, util::store_member_t<TInterpolator&&>>(util::forward<TDiscreteField>(discrete_field), util::forward<TInterpolator>(interpolator)))

} // end of ns field
