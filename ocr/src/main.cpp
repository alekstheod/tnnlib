#include "NeuralNetwork/BackPropagation/BepAlgorithm.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/ActivationFunction/BiopolarSigmoidFunction.h"
#include "NeuralNetwork/ActivationFunction/LogScaleSoftmaxFunction.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/ActivationFunction/SoftmaxFunction.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"
#include "NeuralNetwork/ActivationFunction/ReluFunction.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/NeuralLayer/OpenCL/OpenCLNeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"
#include "NeuralNetwork/BackPropagation/BPAsyncNeuralLayer.h"
#include "NeuralNetwork/Serialization/Cereal.h"

#include <MPL/Tuple.h>

#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#include <boost/filesystem.hpp>
#undef BOOST_SYSTEM_NO_DEPRECATED
#endif

#include <boost/gil/channel_algorithm.hpp>
#include <boost/gil/channel.hpp>
#include <boost/gil.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/io/read_and_convert_image.hpp>

#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>
#include <boost/gil/extension/dynamic_image/any_image.hpp>
#include <boost/gil/extension/dynamic_image/dynamic_image_all.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>

#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

typedef float VarType;

namespace {
    using namespace boost::gil;
    using namespace boost::gil::detail;
    const std::string alphabet("0123456789");
    constexpr std::size_t width = 12;
    constexpr std::size_t height = 15;
    constexpr std::size_t inputsNumber = width * height;
} // namespace


using Perceptron =
 nn::Perceptron< VarType,
                 nn::InputLayer< nn::Neuron, nn::SigmoidFunction, inputsNumber, 1 >,
                 nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 30 >,
                 nn::NeuralLayer< nn::Neuron, nn::SoftmaxFunction, 10 > >;

using InputData = typename Perceptron::Input;

using Algo = nn::bp::BepAlgorithm< Perceptron, nn::bp::CrossEntropyError >;

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
    read_and_convert_image(fileName.c_str(), srcImg, png_tag());

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
            *out = InputData{static_cast< float >(src_it[x]) / 255.f}; // input in a range of (0-1)
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
            cereal::JSONInputArchive ia(file);
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
        std::array< InputData, inputsNumber > inputs = {InputData{}};
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
    cereal::JSONOutputArchive oa(strm);
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
    // static Perceptron tmp = readPerceptron("perceptron.json");
    static Algo algorithm(0.0009f);
    // algorithm.setMemento(tmp.getMemento());

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
            } catch(const std::exception& e) {
                std::cout << "Invalid image found :" << image
                          << " exception: " << e.what() << std::endl;
            }
        }
    }

    auto errorFunc = [](unsigned int epoch, VarType error) {
        std::cout << "Epoch:" << epoch << " error:" << error << std::endl;
        return error > 0.1f;
    };

    static Perceptron perceptron =
     algorithm.calculate(prototypes.begin(), prototypes.end(), errorFunc);

    save(perceptron, "perceptron.json");
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
        std::cout
         << "./ocr [folder] where  [folder] is a directory with your "
            "samples, this command will generate a perceptron.json file"
         << std::endl
         << std::endl;
        std::cout << "./ocr perceptron.json [file] where [file] is a png image "
                     "which has to be recognized"
                  << std::endl
                  << std::endl;
    }

    return result;
}
