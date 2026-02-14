//
// Created by moinshaikh on 2/5/26.
//
#include<torch/torch.h>
#include<torch/nn.h>

#include"../../include/Model/nnBase.hpp"

#include "../../include/Model/modelUtils.hpp"

#include"../third_party/doctest.hpp"

#include"../../include/Model/mlp_base.hpp"

namespace LunarAlighting
{
    /**
     * @brief Constructs the NNBase object.
     *
     * Initializes the neural network base class.
     * - If recurrent is true, initializes the Gated Recurrent Unit (GRU).
     * - Registers the GRU module.
     * - Initializes weights for the GRU parameters.
     *
     * @param recurrent Boolean flag indicating if the network is recurrent.
     * @param recurrentInputSize Size of the input for the recurrent layer.
     * @param hiddenSize Size of the hidden layer.
     */
    NNBase::NNBase(bool recurrent, unsigned int recurrentInputSize, unsigned int hiddenSize) :
     gatedRecurrentUnit(nullptr),
    hiddenSize(hiddenSize),
    recurrent(recurrent)
    {
        // Init GRU
        if (recurrent)
        {
            gatedRecurrentUnit = torch::nn::GRU(torch::nn::GRUOptions(recurrentInputSize, hiddenSize));
            register_module("gatedRecurrentUnit", gatedRecurrentUnit);
            // Init weights
            initWeights(gatedRecurrentUnit->named_parameters(), 1, 0);
        }
    }

    /**
     * @brief Forward pass for the network.
     *
     * This is a placeholder implementation.
     *
     * @param torch::Tensor Input tensor.
     * @param torch::Tensor Recurrent hidden states.
     * @param torch::Tensor Masks.
     * @return std::vector<torch::Tensor> Empty vector.
     */



    /**
     * @brief Gets the hidden size of the network.
     *
     * @return unsigned int The hidden size if recurrent, otherwise 1.
     */
    unsigned int NNBase::getHiddenSize() const
    {
            if (recurrent)
            {
                return hiddenSize;
            }
            return 1;
    }

    /**
     * @brief Forwards inputs through Gated Recurrent Units (GRU).
     *
     * Handles the forward pass of the GRU with support for:
     * - Standard sequential processing.
     * - Flattened input tensors (timesteps * agents).
     * - Masking for episode boundaries.
     *
     * Detailed steps:
     * - Checks if input size matches hidden state size.
     * - If mismatched (flattened input):
     *   - Unflattens input and masks to (timesteps, agents, features).
     *   - Identifies indices where masks are zero (indicating resets).
     *   - Processes segments between resets sequentially.
     *   - Concatenates outputs and reshapes back to flattened format.
     *
     * @param x Input tensor.
     * @param rnn_hxs Recurrent hidden states.
     * @param masks Masks tensor.
     * @return std::vector<torch::Tensor> Vector containing output and new hidden states.
     */
    std::vector<torch::Tensor> NNBase::forwardGatedRecurrentUnits(torch::Tensor x, torch::Tensor rnn_hxs, torch::Tensor masks)
    {
        if (x.size(0) == rnn_hxs.size(0))
        {
           auto [output , state] = gatedRecurrentUnit->forward(x.unsqueeze(0),
                                           (rnn_hxs * masks).unsqueeze(0));
            return {output.squeeze(0), state.squeeze(0)};
        }
        else
        {
            // x is a (timesteps, agents, -1) tensor that has been flattened to
            // (timesteps * agents, -1)
            auto agents = rnn_hxs.size(0);
            auto timesteps = x.size(0) / agents;

            // Unflatten
            x = x.view({timesteps, agents, x.size(1)});

            // Same for masks
            masks = masks.view({timesteps, agents});

            // Figure out which steps in the sequence have a zero for any agent
            // We assume the first timestep has a zero in it
            auto has_zeros = (masks.narrow(0, 1, masks.size(0) - 1) == 0)
                                 .any(-1)
                                 .nonzero()
                                 .squeeze();

            // +1 to correct the masks[1:]
            has_zeros += 1;

            // Add t=0 and t=timesteps to the list
            // has_zeros = [0] + has_zeros + [timesteps]
            has_zeros = has_zeros.contiguous().to(torch::kInt);
            std::vector<int> has_zeros_vec(
                has_zeros.data_ptr<int>(),
                has_zeros.data_ptr<int>() + has_zeros.numel());
            has_zeros_vec.insert(has_zeros_vec.begin(), {0});
            has_zeros_vec.push_back(timesteps);

            rnn_hxs = rnn_hxs.unsqueeze(0);
            std::vector<torch::Tensor> outputs;
            for (unsigned int i = 0; i < has_zeros_vec.size() - 1; ++i)
            {
                // We can now process long runs of timesteps without dones in them in
                // one go
                auto start_idx = has_zeros_vec[i];
                auto end_idx = has_zeros_vec[i + 1];

                auto [output,state] =gatedRecurrentUnit(
                    x.index({torch::arange(start_idx,
                                           end_idx,
                                           torch::TensorOptions(torch::kLong))}).to(torch::kFloat),
                    rnn_hxs * masks[start_idx].view({1, -1, 1}).to(torch::kFloat));

                outputs.push_back(output);
            }

            // x is a (timesteps, agents, -1) tensor
            x = torch::cat(outputs, 0).squeeze(0);
            x = x.view({timesteps * agents, -1});
            rnn_hxs = rnn_hxs.squeeze(0);

            return {x, rnn_hxs};
        }
    }

    TEST_CASE("NNBase")
    {
        auto base = std::make_shared<MlpBase>(5, true, 10);

        SUBCASE("forward_gru() outputs correct shapes when given samples from one"
                " agent")
        {
            auto inputs = torch::rand({4, 5});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = base->forwardGatedRecurrentUnits(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 2);

            // x
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 10);

            // rnn_hxs
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 10);
        }

        SUBCASE("forward_gru() outputs correct shapes when given samples from "
                "multiple agents")
        {
            auto inputs = torch::rand({12, 5});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({12, 1});
            auto outputs = base->forwardGatedRecurrentUnits(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 2);

            // x
            CHECK(outputs[0].size(0) == 12);
            CHECK(outputs[0].size(1) == 10);

            // rnn_hxs
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 10);
        }
    }
}
