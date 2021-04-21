namespace template_tensors {

template <typename T>
__host__ __device__
auto singleton(T&& t)
RETURN_AUTO(SingletonT<typename std::remove_reference<T>::type>(util::forward<T>(t)))

} // end of ns tensor