#ifndef MATH_H
#define MATH_H

namespace utils
{
  template< typename Var >
  Var log( const Var& var );
  
  template< typename Var >
  Var exp( const Var& var );
  
  template< typename Var >
  Var pow( const Var& var, int power );
  
  template< typename Var >
  Var sqrt( const Var& var);
}

#endif