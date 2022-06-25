#pragma once

namespace nn {

    template< typename Func, unsigned int threshold >
    class Threshold {
      private:
        Func m_func;
        static_assert(threshold <= 100, "Invalid template parameter threshold");
        typedef typename Func::Var Var;

      public:
        /**
         * Will calculate the equation
         * for the given input value and apply the throshold to its output.
         * @return the calculation result.
         */
        template< typename Iterator >
        Var calculate(const Var& sum, Iterator begin, Iterator end) const {
            Var t = threshold / Var{100.f};
            return m_func.calculate(sum, begin, end) > t ?
                          Var{1.0f} :
                          Var{0.0f};
        }

        template< typename Iterator >
        Var sum(Iterator begin, Iterator end, const Var& start) const {
            return m_func.sum(begin, end, start);
        }

        Var delta(const Var& output, const Var& expectedOutput) const {
            return m_func.delta(output, expectedOutput);
        }

        Var derivate(const Var& output) const {
            return m_func.derivate(output);
        }

        operator Func() {
            return m_func;
        }

        Func& operator*() {
            return m_func;
        }
    };
} // namespace nn

