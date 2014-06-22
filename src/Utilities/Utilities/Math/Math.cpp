#include <Utilities/Math/Math.h>
#include <cmath>

namespace utils 
{
  template<>
  float log<float>( const float& var ){
    return std::log(var);
  }
  
  template<>
  double log<double>( const double& var ){
    return std::log(var);
  }
  
  template<>
  float exp<float>( const float& var ){
    return std::exp( var );
  }
  
  template<>
  double exp<double>( const double& var ){
    return std::exp( var );
  }
  
  template<>
  float pow<float>( const float& var, int power ){
    return std::pow(var, power);
  }
  
  template<>
  double pow<double>( const double& var, int power ){
    return std::pow(var, power);
  }
  
  template<>
  float sqrt<float>( const float& var){
    return std::sqrt(var);
  }
  
  template<>
  double sqrt<double>( const double& var){
    return std::sqrt(var);
  }
}