/// \file
// Range v3 library
//
//  Copyright Eric Niebler 2014
//
//  Use, modification and distribution is subject to the
//  Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Project home: https://github.com/ericniebler/range-v3
//
#ifndef RANGES_V3_UTILITY_COUNTED_ITERATOR_HPP
#define RANGES_V3_UTILITY_COUNTED_ITERATOR_HPP

#include <utility>
#include <meta/meta.hpp>
#include <range/v3/range_fwd.hpp>
#include <range/v3/utility/iterator.hpp>
#include <range/v3/utility/iterator_traits.hpp>
#include <range/v3/utility/iterator_concepts.hpp>
#include <range/v3/utility/basic_iterator.hpp>

namespace ranges
{
    inline namespace v3
    {
        /// \cond
        namespace detail
        {
            template<typename A, typename B>
            using UnambiguouslyConvertible =
                meta::or_c<
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                    (bool)Same<A, B>::value,
                    ConvertibleTo<A, B>::value == !ConvertibleTo<B, A>::value>;
#else
                    (bool)Same<A, B>(),
                    ConvertibleTo<A, B>() == !ConvertibleTo<B, A>()>;
#endif

            template<typename A, typename B>
            using UnambiguouslyConvertibleType =
                meta::_t<
                    meta::if_c<
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                        (bool)Same<A, B>::value,
                        meta::id<A>,
                        meta::if_c<
                            ConvertibleTo<A, B>::value && !ConvertibleTo<B, A>::value,
                            meta::id<A>,
                            meta::if_c<
                                ConvertibleTo<B, A>::value && !ConvertibleTo<A, B>::value,
#else
                        (bool)Same<A, B>(),
                        meta::id<A>,
                        meta::if_c<
                            ConvertibleTo<A, B>() && !ConvertibleTo<B, A>(),
                            meta::id<A>,
                            meta::if_c<
                                ConvertibleTo<B, A>() && !ConvertibleTo<A, B>(),
#endif
                                meta::id<B>,
                                meta::nil_>>>>;

            template<typename I, typename D /* = iterator_difference_t<I>*/>
            struct counted_cursor
            {
                using single_pass = SinglePass<I>;
                struct mixin
                  : basic_mixin<counted_cursor>
                {
                    mixin() = default;
                    mixin(counted_cursor pos)
                      : basic_mixin<counted_cursor>{std::move(pos)}
                    {}
                    mixin(I it, D n)
                      : mixin(counted_cursor{it, n})
                    {}
                    I base() const
                    {
                        return this->get().base();
                    }
                    D count() const
                    {
                        return this->get().count();
                    }
                };
            private:
                template<typename OtherI, typename OtherD>
                friend struct counted_cursor;
                using iterator_concept_ =
                    concepts::most_refined<meta::list<concepts::Iterator, concepts::WeakIterator>, I>;
                I it_;
                D n_;

                // Overload the advance algorithm for counted_iterators.
                // This is much faster. This gets found by ADL because
                // counted_cursor is an associated type of counted_iterator.
                friend void advance(counted_iterator<I, D> &it, iterator_difference_t<I> n)
                {
                    counted_cursor &cur = get_cursor(it);
                    cur.n_ -= n;
                    ranges::advance(cur.it_, n);
                }
                // Overload uncounted and recounted for packing and unpacking
                // counted iterators
                friend I uncounted(counted_iterator<I, D> i)
                {
                    return i.base();
                }
                friend counted_iterator<I, D>
                recounted(counted_iterator<I, D> const &j, I i, iterator_difference_t<I> n)
                {
                    RANGES_ASSERT(!ForwardIterator<I>() || ranges::next(j.base(), n) == i);
                    return {i, j.count() - n};
                }
            public:
                counted_cursor()
                  : it_{}, n_{}
                {}
                counted_cursor(I it, D n)
                  : it_(std::move(it)), n_(n)
                {}
                template<typename OtherI, typename OtherD,
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                    CONCEPT_REQUIRES_(ConvertibleTo<OtherI, I>::value && ConvertibleTo<OtherD, D>::value)>
#else
                    CONCEPT_REQUIRES_(ConvertibleTo<OtherI, I>() && ConvertibleTo<OtherD, D>())>
#endif
                counted_cursor(counted_cursor<OtherI, OtherD> that)
                  : it_(std::move(that.it_)), n_(std::move(that.n_))
                {}
                auto get() const -> decltype(*it_)
                {
                    return *it_;
                }
                bool done() const
                {
                    return 0 == n_;
                }
                void next()
                {
                    ++it_;
                    --n_;
                }
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                CONCEPT_REQUIRES(EqualityComparable<D>::value)
#else
                CONCEPT_REQUIRES(EqualityComparable<D>())
#endif
                bool equal(counted_cursor const &that) const
                {
                    return n_ == that.n_;
                }
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                CONCEPT_REQUIRES(BidirectionalIterator<I>::value)
#else
                CONCEPT_REQUIRES(BidirectionalIterator<I>())
#endif
                void prev()
                {
                    --it_;
                    ++n_;
                }
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                CONCEPT_REQUIRES(RandomAccessIterator<I>::value)
#else
                CONCEPT_REQUIRES(RandomAccessIterator<I>())
#endif
                void advance(iterator_difference_t<I> n)
                {
                    it_ += n;
                    n_ -= n;
                }
                D distance_to(counted_cursor<I> const &that) const
                {
                    return n_ - that.n_;
                }
                D distance_remaining() const
                {
                    return n_;
                }
                I base() const
                {
                    return it_;
                }
                D count() const
                {
                    return n_;
                }
            };
        }
        /// \endcond

        /// \addtogroup group-utility
        /// @{

#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
        template<typename I, CONCEPT_REQUIRES_(WeakInputIterator<I>::value)>
#else
        template<typename I, CONCEPT_REQUIRES_(WeakInputIterator<I>())>
#endif
        counted_iterator<I> make_counted_iterator(I i, iterator_difference_t<I> n)
        {
            return counted_iterator<I>{std::move(i), n};
        }
        /// @}
    }
}

#endif
