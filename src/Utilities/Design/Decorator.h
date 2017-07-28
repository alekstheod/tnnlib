#ifndef DECORATOR_H
#define DECORATOR_H

#include <functional>
#include <utility>

namespace util {

    using Action = std::function< void() >;

    namespace detail {

        template < typename Internal > class Operation {
            public:
            Operation (Internal& internal, Action before, Action& after) : m_internal (internal), m_after (after) {
                before ();
            }

            ~Operation () {
                m_after ();
            }

            auto operator-> () -> Internal* {
                return &m_internal;
            }

            private:
            Internal& m_internal;
            Action& m_after;
        };
    }

    template < typename Internal > class Decorator {
        public:
        template < typename... Args > Decorator (Args... args) : m_internal (std::forward (args)...) {
        }

        auto operator-> () -> detail::Operation< Internal > {
            return detail::Operation< Internal > (m_internal, m_before, m_after);
        }

        private:
        Internal m_internal;
        Action m_before;
        Action m_after;
    };

    template < typename Internal > auto decorate (const Action& before, const Action& after) -> Decorator< Internal > {
        return Decorator< Internal > (before, after);
    }
}

#endif
