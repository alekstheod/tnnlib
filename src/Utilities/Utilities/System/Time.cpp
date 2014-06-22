#include <Utilities/System/Time.h>
#include <chrono>
#include <random>
#include <boost/numeric/conversion/cast.hpp>

namespace utils
{
namespace priv
{

template<typename Var>
Var rnd( unsigned int maxValue ){
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_int_distribution<int> normalDist(0, maxValue - 1);
    return boost::numeric_cast<Var>( normalDist(engine) );
}

template<>
float randomize<float> ( unsigned int maxValue )
{
    return rnd<float>(maxValue);
}

template<>
int randomize<int> ( unsigned int maxValue )
{
    return rnd<int>(maxValue);
}

template<>
unsigned int randomize<unsigned int> ( unsigned int maxValue )
{
    return rnd<unsigned int>(maxValue);
}

}
}
