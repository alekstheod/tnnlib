#ifndef NN_INPUT_H
#define NN_INPUT_H
#include <boost/serialization/serialization.hpp>
#include <boost/numeric/conversion/cast.hpp>

namespace nn{
  
template<typename Var>
struct Input{
  Input():weight( boost::numeric_cast<Var>(0)  ), value(boost::numeric_cast<Var>(0) ){}
  Input(const Var& w, const Var& v):weight(w), value(v){}
  Var weight;
  Var value;
  
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
      ar & BOOST_SERIALIZATION_NVP(weight);
      ar & BOOST_SERIALIZATION_NVP(value);
  }
  
};
  
}

#endif