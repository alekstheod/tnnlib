#include <NeuralNetwork/Perceptron/Perceptron.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SoftmaxFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/BiopolarSigmoidFunction.h>
#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BepAlgorithm.h>
#include <NeuralNetwork/Neuron/ActivationFunction/LogScaleSoftmaxFunction.h>
//#include <NeuralNetwork/NeuralLayer/OpenCLNeuralLayer.h>
#include <NeuralNetwork/Config.h>
#include <cmath>

#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#include <boost/filesystem.hpp>
#undef BOOST_SYSTEM_NO_DEPRECATED
#endif

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <fstream>
#include <iomanip>
#include <set>

#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

#if defined(NN_CC_MSVC)
# pragma warning(push)
// This function or variable may be unsafe
# pragma warning(disable:4996)
#endif

#include <boost/gil/gil_all.hpp>
#include <boost/gil/channel_algorithm.hpp>
#include <boost/gil/channel.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>		
#include <boost/gil/extension/io/dynamic_io.hpp>
#include <boost/gil/extension/dynamic_image/any_image.hpp>
#include <boost/gil/extension/dynamic_image/dynamic_image_all.hpp>
#include "gil/extension/numeric/sampler.hpp"
#include "gil/extension/numeric/resample.hpp"

#if defined(NN_CC_MSVC)
# pragma warning(pop)
#endif

#include "Var.h"

typedef long double VarType;

using namespace boost::gil;
using namespace boost::gil::detail;
static const std::string alphabet("0123456789");
static const unsigned int width = 49;
static const unsigned int height = 67;
static const unsigned int inputsNumber = width * height;

typedef nn::Perceptron< VarType,
			nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 10 , inputsNumber, 1000 >, 
			nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 80, 10, 1000>, 
			nn::NeuralLayer<nn::Neuron, nn::SoftmaxFunction, 10, 1000>
		       > Perceptron;
		       
typedef nn::Perceptron< VarType,
			nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 20, inputsNumber, 1000>,
			nn::NeuralLayer<nn::Neuron, nn::SoftmaxFunction,  inputsNumber, 80, 1000 >
		      > AutoEncoder;

typedef nn::bp::BepAlgorithm< AutoEncoder, nn::bp::CrossEntropyError> AutoEncAlgo;
typedef nn::bp::BepAlgorithm< Perceptron, nn::bp::CrossEntropyError> Algo;

template <typename Out>
struct halfdiff_cast_channels {
    template <typename T>
    Out operator()(const T& in1, const T& in2) const {
        return Out( (in1-in2)/2 );
    }
};

template <typename SrcView, typename DstView>
void convert_color(const SrcView& src, const DstView& dst) {
    typedef typename channel_type<DstView>::type d_channel_t;
    typedef typename channel_convert_to_unsigned<d_channel_t>::type channel_t;
    typedef pixel<channel_t, gray_layout_t>  gray_pixel_t;

    copy_pixels(color_converted_view<gray_pixel_t>(src), dst);
}

template<typename Iterator>
void readImage(std::string fileName, Iterator out) {
    using namespace boost::gil;
    using namespace nn;
    using namespace nn::bp;

    rgb8_image_t srcImg;
    png_read_image(fileName.c_str(), srcImg);

    gray8_image_t dstImg(srcImg.dimensions());
    gray8_pixel_t white(255);
    fill_pixels(view(dstImg), white);

    gray8_image_t grayImage(srcImg.dimensions());
    convert_color( view(srcImg), view(grayImage) );
    auto grayView = view(grayImage);

    gray8_image_t scaledImage(width, height);
    resize_view(grayView, view(scaledImage), bilinear_sampler() );
    auto srcView = view(scaledImage);

    for (int y=0; y<srcView.height(); ++y) {
        gray8c_view_t::x_iterator src_it( srcView.row_begin(y) );
        for (int x=0; x<srcView.width(); ++x) {
            *out = src_it[x]/255.f;// < 130? 1.f: -1.f;//255.f;
            out++;
        }
    }
}

Perceptron readPerceptron(std::string fileName) {
    Perceptron perceptron;
    if( boost::filesystem::exists(fileName.c_str()) ){
      std::ifstream file(fileName);
      if( file.good() ) {
	  Perceptron::Memento memento;
	  boost::archive::xml_iarchive ia ( file );
	  ia  >> BOOST_SERIALIZATION_NVP ( memento );

	  perceptron.setMemento ( memento );
      } else {
	  throw nn::NNException("Invalid perceptron file name", __FILE__, __LINE__);
      }
    }
    
    return perceptron;
}

void recognize(std::string perceptron, std::string image) {
    try {
        std::array<VarType, inputsNumber> inputs = {0};
        readImage(image, inputs.begin() );
        std::vector<VarType> result(alphabet.length(), VarType(0.f) );
        readPerceptron(perceptron).calculate(inputs.begin(), inputs.end(), result.begin() );
        for( unsigned int i = 0; i < result.size(); i++ ) {
	  std::cout << "Symbol: " << 
		       alphabet[i] << 
		       " " <<
		       result[i] <<
		       std::endl;
        }
        
    } catch( const nn::NNException& e) {
        std::cout << e.what() << std::endl;
    } catch( const std::exception& e) {
        std::cout << e.what() << std::endl;
    } catch(...) {
        std::cout << "Unknown error" << std::endl;
    }
}

template<typename Perc>
void save(const Perc& perc, std::string name)
{
    typename Perc::Memento memento = perc.getMemento();
    std::ofstream strm( name );
    boost::archive::xml_oarchive oa ( strm );
    oa << BOOST_SERIALIZATION_NVP ( memento );
    strm.flush();
}

template<typename Files>
AutoEncoder calculateAutoEncoder(Files files){
    std::cout << "AutoEncoder calculation started" << std::endl;
    std::vector<AutoEncAlgo::Prototype> prototypes;
    for( auto image : files) {
        if( !boost::filesystem::is_directory( image ) ) {
            try {
                AutoEncAlgo::Prototype proto;
                readImage( image, std::get<0>(proto).begin() );
		std::copy( std::get<0>(proto).begin(),
			   std::get<0>(proto).end(),
			   std::get<1>(proto).begin());
		
                prototypes.push_back(proto);
            } catch(const std::exception&) {
                std::cout << "Invalid image found :" << image << std::endl;
            }
        }
    }
    
    static AutoEncAlgo autoEncAlgo(0.005f, 0.01f);
    
    static std::size_t counter = 0;
    static AutoEncoder enc = autoEncAlgo.calculate(prototypes.begin(), 
							  prototypes.end(), 
							  [](VarType error) {
							      counter++;
							      if(counter > 0){
								counter = 0;
								std::cout << error << std::endl;
							      }
							  });  
    
    save(enc, "autoencoder.xml");
}

void calculateWeights(std::string imagesPath) {
    using namespace boost::filesystem;
    path directory(imagesPath);
    directory_iterator end_iter;

    std::vector< std::string > files;
    if ( exists(directory) && is_directory(directory))
    {
        for( directory_iterator dir_iter(directory) ; dir_iter != end_iter ; ++dir_iter)
        {
            if (is_regular_file(dir_iter->status()) )
            {
                files.push_back( dir_iter->path().string() );
            }
        }
    }
    
    //static AutoEncoder autoEnc = calculateAutoEncoder(files);
    
    std::cout << "Perceptron calculation started" << std::endl;
    static Perceptron tmp = readPerceptron("perceptron.xml");
    static Algo algorithm (0.003f, 0.001f );
    algorithm.setMemento( tmp.getMemento() );
    
    std::vector<Algo::Prototype> prototypes;
    for( auto image : files ) {
        if( !boost::filesystem::is_directory( image ) ) {
            try {
                Algo::Prototype proto;
                readImage( image, std::get<0>(proto).begin() );
                std::fill(std::get<1>(proto).begin(), std::get<1>(proto).end(), 0.f);
                char ch = path(image).filename().string()[0];
                size_t pos = alphabet.find(ch);
                std::get<1>(proto)[pos] = 1.0f;
                prototypes.push_back(proto);
            } catch(const std::exception&) {
                std::cout << "Invalid image found :" << image << std::endl;
            }
        }
    }

    static std::size_t counter = 0;
    auto errorFunc = [](VarType error) {counter++;
					if(counter > 1000){
					  counter = 0;
					  std::cout << error << std::endl;
					}};
					
    static Perceptron perceptron = algorithm.calculate(prototypes.begin(), 
						       prototypes.end(), 
						       errorFunc);

    save(perceptron, "perceptron.xml");
}


int main(int argc, char** argv)
{
    int result = -1;
    if( argc == 3 ) {
        recognize(argv[1], argv[2]);
        result = 0;
    } else if( argc == 2 ) {
        calculateWeights(argv[1]);
        result = 0;
    } else {
        std::cout << std::endl << "Usage : " << std::endl << std::endl;
        std::cout << "./ocr [folder] where  [folder] is a directory with your samples, this command will generate a perceptron.xml file" << std::endl << std::endl;
        std::cout << "./ocr perceptron.xml [file] where [file] is a png image which has to be recognized" << std::endl << std::endl;
    }
  
    return result;
}


