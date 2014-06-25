#include <NeuralNetwork/Perceptron/Perceptron.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Perceptron/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SoftmaxFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/BiopolarSigmoidFunction.h>
#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BepAlgorithm.h>
#include <NeuralNetwork/Neuron/ActivationFunction/LogScaleSoftmaxFunction.h>
#include <NeuralNetwork/Config.h>

#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#include <boost/filesystem.hpp>
#undef BOOST_SYSTEM_NO_DEPRECATED
#endif

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <fstream>
#include <iomanip>

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


typedef float VarType;

using namespace boost::gil;
using namespace boost::gil::detail;
static const std::string alphabet("0123456789");
static const unsigned int width = 10;
static const unsigned int height = 14;
static const unsigned int inputsNumber = width * height;

typedef nn::Perceptron< float,
			nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, inputsNumber>, 
			nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 30>, 
			nn::NeuralLayer<nn::Neuron, nn::SoftmaxFunction, 10>
		       > Perceptron;

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

template<typename InputIterator>
void readImage(std::string fileName, InputIterator input) {
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
            *input = src_it[x] < 130 ? 1.f : -1.f;
            input++;
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
        bool found = false;
        int foundedSymbols = 0;
        int position = 0;
        for( unsigned int i = 0; i < result.size(); i++ ) {
            if( result[i] > 0.8f ) {
                position = i;
                found = true;
                foundedSymbols++;
            }
        }

        if( !found || foundedSymbols > 1 ) {
            std::cout << "Unknown symbol" << std::endl;
        } else {
            std::cout << "Found symbol: "<< alphabet[position] << std::endl;
        }
    } catch( const nn::NNException& e) {
        std::cout << e.what() << std::endl;
    } catch( const std::exception& e) {
        std::cout << e.what() << std::endl;
    } catch(...) {
        std::cout << "Unknown error" << std::endl;
    }
}

void calculateWeights(std::string imagesPath) {
    using namespace boost::filesystem;
    path directory(imagesPath);
    directory_iterator end_iter;

    std::set< std::string > files;
    if ( exists(directory) && is_directory(directory))
    {
        for( directory_iterator dir_iter(directory) ; dir_iter != end_iter ; ++dir_iter)
        {
            if (is_regular_file(dir_iter->status()) )
            {
                files.insert( dir_iter->path().string() );
            }
        }
    }

    Perceptron tmp = readPerceptron("perceptron.xml");
    Algo algorithm (0.04f, 0.01f );
    algorithm.setMemento( tmp.getMemento() );
    
    std::vector<Algo::Prototype> prototypes;
    for( auto i = files.begin(); i != files.end(); i++ ) {
        if( !boost::filesystem::is_directory( *i ) ) {
            try {
                Algo::Prototype proto;
                readImage( *i, std::get<0>(proto).begin() );
                std::fill(std::get<1>(proto).begin(), std::get<1>(proto).end(), 0.f);
                char ch = path(*i).filename().string()[0];
                size_t pos = alphabet.find(ch);
                std::get<1>(proto)[pos] = 1.0f;
                prototypes.push_back(proto);
            } catch(const std::exception&) {
                std::cout << "Invalid image found :" << *i << std::endl;
            }
        }
    }

    Perceptron perceptron = algorithm.calculatePerceptron(prototypes.begin()
                            , prototypes.end()
    , [](VarType error) {
        std::cout << error << std::endl;
    });

    Perceptron::Memento memento = perceptron.getMemento();
    std::ofstream strm( "perceptron.xml");
    boost::archive::xml_oarchive oa ( strm );
    oa << BOOST_SERIALIZATION_NVP ( memento );
    strm.flush();
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


