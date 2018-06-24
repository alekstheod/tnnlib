/// \file
// Range v3 library
//
//  Copyright Eric Niebler 2013-2014
//
//  Use, modification and distribution is subject to the
//  Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Project home: https://github.com/ericniebler/range-v3
//

#ifndef RANGES_V3_ACTION_STABLE_SORT_HPP
#define RANGES_V3_ACTION_STABLE_SORT_HPP

#include <functional>
#include <range/v3/range_fwd.hpp>
#include <range/v3/action/action.hpp>
#include <range/v3/algorithm/stable_sort.hpp>
#include <range/v3/utility/functional.hpp>
#include <range/v3/utility/iterator_concepts.hpp>
#include <range/v3/utility/iterator_traits.hpp>
#include <range/v3/utility/static_const.hpp>

namespace ranges
{
    inline namespace v3
    {
        /// \addtogroup group-actions
        /// @{
        namespace action
        {
            struct stable_sort_fn
            {
            private:
                friend action_access;
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                template<typename C, typename P = ident, CONCEPT_REQUIRES_(!Range<C>::value)>
#else
                template<typename C, typename P = ident, CONCEPT_REQUIRES_(!Range<C>())>
#endif
                static auto bind(stable_sort_fn stable_sort, C pred, P proj = P{})
                RANGES_DECLTYPE_AUTO_RETURN
                (
                    std::bind(stable_sort, std::placeholders::_1, protect(std::move(pred)),
                        protect(std::move(proj)))
                )
            public:
                struct ConceptImpl
                {
                    template<typename Rng, typename C = ordered_less, typename P = ident,
                        typename I = range_iterator_t<Rng>>
                    auto requires_(Rng&&, C&& = C{}, P&& = P{}) -> decltype(
                        concepts::valid_expr(
                            concepts::model_of<concepts::ForwardRange, Rng>(),
                            concepts::is_true(Sortable<I, C, P>())
                        ));
                };

                template<typename Rng, typename C = ordered_less, typename P = ident>
                using Concept = concepts::models<ConceptImpl, Rng, C, P>;

                template<typename Rng, typename C = ordered_less, typename P = ident,
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                    CONCEPT_REQUIRES_(Concept<Rng, C, P>::value)>
#else
                    CONCEPT_REQUIRES_(Concept<Rng, C, P>())>
#endif
                Rng operator()(Rng && rng, C pred = C{}, P proj = P{}) const
                {
                    ranges::stable_sort(rng, std::move(pred), std::move(proj));
                    return std::forward<Rng>(rng);
                }

            #ifndef RANGES_DOXYGEN_INVOKED
                template<typename Rng, typename C = ordered_less, typename P = ident,
#ifdef RANGES_WORKAROUND_MSVC_SFINAE_CONSTEXPR
                    CONCEPT_REQUIRES_(!Concept<Rng, C, P>::value)>
#else
                    CONCEPT_REQUIRES_(!Concept<Rng, C, P>())>
#endif
                void operator()(Rng &&, C && = C{}, P && = P{}) const
                {
                    CONCEPT_ASSERT_MSG(ForwardRange<Rng>(),
                        "The object on which action::stable_sort operates must be a model of the "
                        "ForwardRange concept.");
                    using I = range_iterator_t<Rng>;
                    CONCEPT_ASSERT_MSG(IndirectCallable<P, I>(),
                        "The projection function must accept objects of the iterator's value type, "
                        "reference type, and common reference type.");
                    CONCEPT_ASSERT_MSG(IndirectCallableRelation<C, Projected<I, P>>(),
                        "The comparator passed to action::stable_sort must accept objects returned "
                        "by the projection function, or of the range's value type if no projection "
                        "is specified.");
                    CONCEPT_ASSERT_MSG(Permutable<I>(),
                        "The iterator type of the range passed to action::stable_sort must allow its "
                        "elements to be permuted; that is, the values must be movable and the "
                        "iterator must be mutable.");
                }
            #endif
            };

            /// \ingroup group-actions
            /// \relates stable_sort_fn
            /// \sa action
            namespace
            {
                constexpr auto&& stable_sort = static_const<action<stable_sort_fn>>::value;
            }
        }
        /// @}
    }
}

#endif
