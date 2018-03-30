#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BepAlgorithm.h>
#include <NeuralNetwork/NeuralLayer/ConvolutionLayer.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/BiopolarSigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/LogScaleSoftmaxFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SoftmaxFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/ReluFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Perceptron/Perceptron.h>
//#include <NeuralNetwork/NeuralLayer/OpenCLNeuralLayer.h>
#include <NeuralNetwork/Config.h>

#include <Utilities/MPL/Tuple.h>

#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#include <boost/filesystem.hpp>
#undef BOOST_SYSTEM_NO_DEPRECATED
#endif

#define png_infopp_NULL (png_infopp) NULL
#define int_p_NULL (int*)NULL

#if defined(NN_CC_MSVC)
#pragma warning(push)
// This function or variable may be unsafe
#pragma warning(disable : 4996)
#endif

#include <boost/gil/channel_algorithm.hpp>
#include <boost/gil/channel.hpp>
#include <boost/gil/extension/dynamic_image/any_image.hpp>
#include <boost/gil/extension/dynamic_image/dynamic_image_all.hpp>
#include <boost/gil/extension/io/dynamic_io.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/image.hpp>

#include "gil/extension/numeric/sampler.hpp"
#include "gil/extension/numeric/resample.hpp"

#include <cereal/archives/xml.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>

#if defined(NN_CC_MSVC)
#pragma warning(pop)
#endif

#include "Var.h"

#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

typedef long double VarType;

namespace {
    using namespace boost::gil;
    using namespace boost::gil::detail;
    const std::string alphabet("0123456789");
    constexpr std::size_t width = 49;
    constexpr std::size_t height = 67;
    constexpr std::size_t inputsNumber = width * height;
    constexpr std::size_t margin = 15;
    constexpr std::size_t stride = 10;
} // namespace


using ConvolutionGrid =
 typename nn::ConvolutionGrid< width, height, stride, margin >::define;

using Perceptron =
 nn::Perceptron< VarType,
                 nn::ConvolutionLayer< nn::NeuralLayer, nn::Neuron, nn::ReluFunction, inputsNumber, ConvolutionGrid >,
                 nn::NeuralLayer< nn::Neuron, nn::SoftmaxFunction, 10, 1000 > >;

using Algo = nn::bp::BepAlgorithm< Perceptron, nn::bp::CrossEntropyError >;

template< typename Out >
struct halfdiff_cast_channels {
    template< typename T >
    Out operator()(const T& in1, const T& in2) const {
        return Out((in1 - in2) / 2);
    }
};

template< typename SrcView, typename DstView >
void convert_color(const SrcView& src, const DstView& dst) {
    typedef typename channel_type< DstView >::type d_channel_t;
    typedef typename channel_convert_to_unsigned< d_channel_t >::type channel_t;
    typedef pixel< channel_t, gray_layout_t > gray_pixel_t;

    copy_pixels(color_converted_view< gray_pixel_t >(src), dst);
}

template< typename Iterator >
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
    convert_color(view(srcImg), view(grayImage));
    auto grayView = view(grayImage);

    gray8_image_t scaledImage(width, height);
    resize_view(grayView, view(scaledImage), bilinear_sampler());
    auto srcView = view(scaledImage);

    for(int y = 0; y < srcView.height(); ++y) {
        gray8c_view_t::x_iterator src_it(srcView.row_begin(y));
        for(int x = 0; x < srcView.width(); ++x) {
            *out = src_it[x];
            out++;
        }
    }
}

Perceptron readPerceptron(std::string fileName) {
    Perceptron perceptron;
    if(boost::filesystem::exists(fileName.c_str())) {
        std::ifstream file(fileName);
        if(file.good()) {
            Perceptron::Memento memento;
            cereal::XMLInputArchive ia(file);
            ia >> memento;

            perceptron.setMemento(memento);
        } else {
            throw std::logic_error("Invalid perceptron file name");
        }
    }

    return perceptron;
}

void recognize(std::string perceptron, std::string image) {
    try {
        std::array< VarType, inputsNumber > inputs = {0};
        readImage(image, inputs.begin());
        std::vector< VarType > result(alphabet.length(), VarType(0.f));
        readPerceptron(perceptron)
         .calculate(inputs.begin(), inputs.end(), result.begin());
        for(unsigned int i = 0; i < result.size(); i++) {
            std::cout << "Symbol: " << alphabet[i] << " " << result[i] << std::endl;
        }
    } catch(const std::exception& e) {
        std::cout << e.what() << std::endl;
    } catch(...) {
        std::cout << "Unknown error" << std::endl;
    }
}

template< typename Perc >
void save(const Perc& perc, std::string name) {
    typename Perc::Memento memento = perc.getMemento();
    std::ofstream strm(name);
    cereal::XMLOutputArchive oa(strm);
    oa << memento;
    strm.flush();
}

void calculateWeights(std::string imagesPath) {
    using namespace boost::filesystem;
    path directory(imagesPath);
    directory_iterator end_iter;

    std::vector< std::string > files;
    if(exists(directory) && is_directory(directory)) {
        for(directory_iterator dir_iter(directory); dir_iter != end_iter; ++dir_iter) {
            if(is_regular_file(dir_iter->status())) {
                files.push_back(dir_iter->path().string());
            }
        }
    }

    std::cout << "Perceptron calculation started" << std::endl;
    static Perceptron tmp = readPerceptron("perceptron.xml");
    static Algo algorithm(0.003f);
    algorithm.setMemento(tmp.getMemento());

    std::vector< Algo::Prototype > prototypes;
    for(auto image : files) {
        if(!boost::filesystem::is_directory(image)) {
            try {
                Algo::Prototype proto;
                readImage(image, std::get< 0 >(proto).begin());
                std::fill(std::get< 1 >(proto).begin(), std::get< 1 >(proto).end(), 0.f);
                char ch = path(image).filename().string()[0];
                size_t pos = alphabet.find(ch);
                std::get< 1 >(proto)[pos] = 1.0f;
                prototypes.push_back(proto);
            } catch(const std::exception&) {
                std::cout << "Invalid image found :" << image << std::endl;
            }
        }
    }

    auto errorFunc = [](unsigned int epoch, VarType error) {
        // if(epoch % 100 == 0) {
        std::cout << "Epoch:" << epoch << " error:" << error << std::endl;
        // }

        return error > 0.001f;
    };

    static Perceptron perceptron =
     algorithm.calculate(prototypes.begin(), prototypes.end(), errorFunc);

    save(perceptron, "perceptron.xml");
}

int main(int argc, char** argv) {
    int result = -1;
    if(argc == 3) {
        recognize(argv[1], argv[2]);
        result = 0;
    } else if(argc == 2) {
        calculateWeights(argv[1]);
        result = 0;
    } else {
        std::cout << std::endl << "Usage : " << std::endl << std::endl;
        std::cout << "./ocr [folder] where  [folder] is a directory with your "
                     "samples, this command will generate a perceptron.xml file"
                  << std::endl
                  << std::endl;
        std::cout << "./ocr perceptron.xml [file] where [file] is a png image "
                     "which has to be recognized"
                  << std::endl
                  << std::endl;
    }

    return result;
}
