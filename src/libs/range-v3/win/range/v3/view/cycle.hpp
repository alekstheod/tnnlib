/// \file cycle.hpp
// Range v3 library
//
//  Copyright Eric Niebler 2013-2015
//  Copyright Gonzalo Brito Gadeschi 2015
//  Copyright Casey Carter 2015
//
//  Use, modification and distribution is subject to the
//  Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Project home: https://github.com/ericniebler/range-v3
//

#ifndef RANGES_V3_VIEW_CYCLE_HPP
#define RANGES_V3_VIEW_CYCLE_HPP

#include <utility>
#include <type_traits>
#include <meta/meta.hpp>
#include <range/v3/range_fwd.hpp>
#include <range/v3/size.hpp>
#include <range/v3/distance.hpp>
#include <range/v3/begin_end.hpp>
#include <range/v3/range_traits.hpp>
#include <range/v3/range_concepts.hpp>
#include <range/v3/view_facade.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/view.hpp>
#include <range/v3/utility/box.hpp>
#include <range/v3/utility/get.hpp>
#include <range/v3/utility/optional.hpp>
#include <range/v3/utility/iterator.hpp>
#include <range/v3/utility/static_const.hpp>

namespace ranges
{
    inline namespace v3
    {
        /// \cond
        namespace detail
        {
            template<typename Rng>
            using cycle_end_ =
                meta::if_<
                    BoundedRange<Rng>,
                    meta::nil_,
                    box<optional<range_iterator_t<Rng>>, end_tag>>;
        }
        /// \endcond

        /// \addtogroup group-views
        ///@{
        template<typename Rng>
        struct cycled_view
          : view_facade<cycled_view<Rng>, infinite>
          , private detail::cycle_end_<Rng>
        {
        private:
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
            CONCEPT_ASSERT(ForwardRange<Rng>::value);
#else
            CONCEPT_ASSERT(ForwardRange<Rng>());
#endif
            friend range_access;
            Rng rng_;

            void dirty_(std::true_type) const
            {}
            void dirty_(std::false_type)
            {
                ranges::get<end_tag>(*this).reset();
            }

            template<bool IsConst>
            struct cursor
            {
            private:
                template<typename T>
                using constify_if = meta::apply<meta::add_const_if_c<IsConst>, T>;
                using cycled_view_t = constify_if<cycled_view>;
                using difference_type_ = range_difference_t<Rng>;
                using iterator = range_iterator_t<constify_if<Rng>>;

                cycled_view_t *rng_;
                iterator it_;

                iterator get_end_(std::true_type, bool = false) const
                {
                    return ranges::end(rng_->rng_);
                }
                template<bool CanBeEmpty = false>
                iterator get_end_(std::false_type, meta::bool_<CanBeEmpty> = {}) const
                {
                    auto &end_ = ranges::get<end_tag>(*rng_);
                    RANGES_ASSERT(CanBeEmpty || end_);
                    if(CanBeEmpty && !end_)
                        end_ = ranges::next(it_, ranges::end(rng_->rng_));
                    return *end_;
                }
                void set_end_(std::true_type) const
                {}
                void set_end_(std::false_type) const
                {
                    auto &end_ = ranges::get<end_tag>(*rng_);
                    if(!end_)
                        end_ = it_;
                }
            public:
                cursor()
                  : rng_{}, it_{}
                {}
                explicit cursor(cycled_view_t &rng)
                  : rng_(&rng), it_(ranges::begin(rng.rng_))
                {}
                constexpr bool done() const
                {
                    return false;
                }
                auto get() const
                RANGES_DECLTYPE_AUTO_RETURN_NOEXCEPT
                (
                    *it_
                )
                bool equal(cursor const &pos) const
                {
                    RANGES_ASSERT(rng_ == pos.rng_);
                    return it_ == pos.it_;
                }
                void next()
                {
                    auto const end = ranges::end(rng_->rng_);
                    RANGES_ASSERT(it_ != end);
                    if(++it_ == end)
                    {
                        this->set_end_(BoundedRange<Rng>());
                        it_ = ranges::begin(rng_->rng_);
                    }
                }
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                CONCEPT_REQUIRES(BidirectionalRange<Rng>::value)
#else
                CONCEPT_REQUIRES(BidirectionalRange<Rng>())
#endif
                void prev()
                {
                    if(it_ == ranges::begin(rng_->rng_))
                        it_ = this->get_end_(BoundedRange<Rng>());
                    --it_;
                }
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                CONCEPT_REQUIRES(RandomAccessRange<Rng>::value)
#else
                CONCEPT_REQUIRES(RandomAccessRange<Rng>())
#endif
                void advance(difference_type_ n)
                {
                    auto const begin = ranges::begin(rng_->rng_);
                    auto const end = this->get_end_(BoundedRange<Rng>(), meta::bool_<true>());
                    auto const d = end - begin;
                    auto const off = ((it_ - begin) + n) % d;
                    it_ = begin + (off < 0 ? off + d : off);
                }
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                CONCEPT_REQUIRES(SizedIteratorRange<iterator, iterator>::value)
#else
                CONCEPT_REQUIRES(SizedIteratorRange<iterator, iterator>())
#endif
                difference_type_ distance_to(cursor const &that) const
                {
                    RANGES_ASSERT(that.rng_ == rng_);
                    return that.it_ - it_;
                }
            };

            cursor<false> begin_cursor()
            {
                return cursor<false>{*this};
            }
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
            CONCEPT_REQUIRES(BoundedRange<Rng const>::value)
#else
            CONCEPT_REQUIRES(BoundedRange<Rng const>())
#endif
            cursor<true> begin_cursor() const
            {
                return cursor<true>{*this};
            }

        public:
            cycled_view() = default;
            cycled_view(cycled_view &&that)
              : detail::cycle_end_<Rng>{}
              , rng_(std::move(that.rng_))
            {}
            cycled_view(cycled_view const &that)
              : detail::cycle_end_<Rng>{}
              , rng_(that.rng_)
            {}
            /// \pre <tt>distance(rng) != 0</tt>
            explicit cycled_view(Rng rng)
              : detail::cycle_end_<Rng>{}
              , rng_(std::move(rng))
            {
                RANGES_ASSERT(ranges::distance(rng) != 0);
            }
            cycled_view& operator=(cycled_view &&that)
            {
                rng_ = std::move(that.rng_);
                this->dirty_(BoundedRange<Rng>{});
                return *this;
            }
            cycled_view& operator=(cycled_view const &that)
            {
                rng_ = that.rng_;
                this->dirty_(BoundedRange<Rng>{});
                return *this;
            }
        };

        namespace view
        {
            struct cycle_fn
            {
            private:
                friend view_access;
                template<class T>
                using Concept = ForwardRange<T>;

            public:
                /// \pre <tt>distance(rng) != 0</tt>
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                template<typename Rng, CONCEPT_REQUIRES_(Concept<Rng>::value)>
#else
                template<typename Rng, CONCEPT_REQUIRES_(Concept<Rng>())>
#endif
                cycled_view<all_t<Rng>> operator()(Rng &&rng) const
                {
                    return cycled_view<all_t<Rng>>{all(std::forward<Rng>(rng))};
                }

#ifndef RANGES_DOXYGEN_INVOKED
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                template<typename Rng, CONCEPT_REQUIRES_(!Concept<Rng>::value)>
#else
                template<typename Rng, CONCEPT_REQUIRES_(!Concept<Rng>())>
#endif
                void operator()(Rng &&) const
                {
                    CONCEPT_ASSERT_MSG(ForwardRange<Rng>(),
                        "The object on which view::cycle operates must be a "
                        "model of the ForwardRange concept.");
                }
#endif
            };

            /// \relates cycle_fn
            /// \ingroup group-views
            namespace
            {
                constexpr auto &&cycle = static_const<view<cycle_fn>>::value;
            }

       } // namespace view
       /// @}
    } // namespace v3
} // namespace ranges

#endif
