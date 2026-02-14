//
// Created by moinshaikh on 2/6/26.
//

#include<memory>

#include<torch/torch.h>

#include"../../include/Model/OutputLayers.hpp"
#include"../../include/Model/modelUtils.hpp"
#include"../../include/Distribution/Distribution.hpp"
#include"../../include/Distribution/Burnoulli.hpp"
#include"../../include/Distribution/Categorical.hpp"
#include"../../include/Distribution/Normal.hpp"

#include"../third_party/doctest.hpp"

/**
 * @namespace LunarAlighting
 * @brief Namespace for the Lunar Alighting project components.
 */
namespace LunarAlighting
{
    BernoulliOutput::BernoulliOutput(
        unsigned int numInputs,
        unsigned int numOutputs) :
    linear(numInputs,numOutputs)
    {
        register_module("linear",linear);
        initWeights(linear->named_parameters(),0.01,0);
    }

    std::unique_ptr<Distribution> BernoulliOutput::forward(torch::Tensor x)
    {
        x = linear(x);
        return std::make_unique<Bernoulli>(nullptr,&x);
    }

    CategoricalOutput::CategoricalOutput(unsigned int numInputs, unsigned int numOutputs) :
    linear(numInputs,numOutputs)
    {
        register_module("linear",linear);
        initWeights(linear->named_parameters(),0.01,0);
    }

    std::unique_ptr<Distribution> CategoricalOutput::forward(torch::Tensor x)
    {
        x = linear(x);
        return std::make_unique<Categorical>(nullptr,&x);
    }

    NormalOutput::NormalOutput(unsigned int num_inputs, unsigned int num_outputs) : linear_loc(num_inputs,num_outputs)
    {

            register_module("linear_loc",linear_loc);
            scale_log = register_parameter("scale_log",torch::zeros({num_outputs}));
            initWeights(linear_loc->named_parameters(),1,0);
    }

    std::unique_ptr<Distribution> NormalOutput::forward(torch::Tensor x)
    {
        auto loc = linear_loc(x);
        auto scale = scale_log.exp();
        return std::make_unique<Normal>(loc, scale);
    }
    TEST_CASE("BernoulliOutput")
    {
        auto output_layer = BernoulliOutput(3, 5);

        SUBCASE("Output distribution has correct output shape")
        {
            float input_array[2][3] = {{0, 1, 2}, {3, 4, 5}};
            auto input_tensor = torch::from_blob(input_array,
                                                 {2, 3},
                                                 torch::TensorOptions(torch::kFloat));
            auto dist = output_layer.forward(input_tensor);

            auto output = dist->sample();

            CHECK(output.sizes().vec() == std::vector<int64_t>{2, 5});
        }
    }

    TEST_CASE("CategoricalOutput")
    {
        auto output_layer = CategoricalOutput(3, 5);

        SUBCASE("Output distribution has correct output shape")
        {
            float input_array[2][3] = {{0, 1, 2}, {3, 4, 5}};
            auto input_tensor = torch::from_blob(input_array,
                                                 {2, 3},
                                                 torch::TensorOptions(torch::kFloat));
            auto dist = output_layer.forward(input_tensor);

            auto output = dist->sample();

            CHECK(output.sizes().vec() == std::vector<int64_t>{2});
        }
    }

    TEST_CASE("NormalOutput")
    {
        auto output_layer = NormalOutput(3, 5);

        SUBCASE("Output distribution has correct output shape")
        {
            float input_array[2][3] = {{0, 1, 2}, {3, 4, 5}};
            auto input_tensor = torch::from_blob(input_array,
                                                 {2, 3},
                                                 torch::TensorOptions(torch::kFloat));
            auto dist = output_layer.forward(input_tensor);

            auto output = dist->sample();

            CHECK(output.sizes().vec() == std::vector<int64_t>{2, 5});
        }
    }
}
