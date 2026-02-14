//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_OUTPUTLAYERS_HPP
#define LUNARALIGHTINGRL_OUTPUTLAYERS_HPP

#include<torch/torch.h>
#include<torch/nn.h>
#include<memory>

#include"../Distribution/Distribution.hpp"

namespace LunarAlighting {
    /**
     * @class OutputLayer
     * @brief Abstract base class for policy output layers in reinforcement learning models.
     *
     * OutputLayer serves as an abstract interface for different types of distribution output layers.
     * It inherits from PyTorch's nn::Module to integrate with PyTorch's neural network framework.
     *
     * This class defines the contract that all concrete output layer implementations must follow.
     * Different RL algorithms may require different distribution types (Bernoulli, Categorical,
     * Normal), so this abstract base allows for polymorphic behavior and extensibility.
    */
    class OutputLayer : public torch::nn::Module
    {
    public:
        /**
         * @brief Virtual destructor for proper cleanup of derived classes.
         *
         * Pure virtual destructor ensures this is an abstract class and cannot be
         * instantiated directly. It serves as the destructor for all derived output layer types.
        */
        virtual ~OutputLayer()= 0;
        /**
          * @brief Computes the distribution output from a tensor input.
          *
          * @param x Input tensor containing the features/activations from the neural network.
          * @return std::unique_ptr<Distribution> A unique pointer to a Distribution object
          *         that encapsulates the probability distribution for the action space.
          *
          * This pure virtual method must be implemented by all derived classes to define
          * how input features are transformed into a specific probability distribution.
          */
        virtual std::unique_ptr<Distribution> forward(torch::Tensor x) = 0;

    };

    inline OutputLayer::~OutputLayer()  {

    }

    /**
     * @class BernoulliOutput
     * @brief Output layer that produces Bernoulli distributions for binary action spaces.
     *
     * BernoulliOutput is used in reinforcement learning for environments with discrete binary actions
     * (e.g., action 0 or 1, go left or go right). It transforms input features into probabilities
     * of taking the positive action via a Bernoulli distribution.
     *
     * The output layer contains a linear transformation that maps input features to a single
     * logit value, which is then converted to a probability for the Bernoulli distribution.
    */

    class BernoulliOutput : public OutputLayer
    {
    private:
        /**
         * @brief Linear layer for feature transformation.
         *
         * Maps from num_inputs features to a single scalar logit value that parameterizes
         * the Bernoulli distribution (probability of the positive action).
         */
        torch::nn::Linear linear;

    public:
        /**
         * @brief Constructor for BernoulliOutput layer.
         *
         * @param numInputs Number of input features from the previous network layer.
         * @param numOutputs Number of discrete actions (typically 2 for binary actions,
         *                   though the actual output is a single probability value).
         */
        BernoulliOutput(unsigned int numInputs, unsigned int numOutputs);

        /**
         * @brief Computes Bernoulli distribution from input tensor.
         *
         * @param x Input tensor of shape (batch_size, num_inputs) containing network features.
         * @return std::unique_ptr<Distribution> A Bernoulli distribution that encodes the
         *         probability of taking the positive action for each sample in the batch.
         */
        std::unique_ptr<Distribution> forward(torch::Tensor x) override;
    };

    /**
        * @class CategoricalOutput
        * @brief Output layer that produces Categorical distributions for discrete action spaces.
        *
        * CategoricalOutput is used for environments with multiple discrete action choices
        * (e.g., move left, right, up, down). It transforms input features into a probability
        * distribution over all possible actions via a Categorical distribution.
        *
        * The output layer contains a linear transformation that maps input features to logits
        * for each action, which are then converted to action probabilities via softmax.
       */
    class CategoricalOutput : public OutputLayer
    {
    private:
        /**
         * @brief Linear layer for feature transformation.
         *
         * Maps from num_inputs features to num_outputs logit values, one for each possible action.
         * These logits are later converted to probabilities via softmax in the Categorical distribution.
        */
        torch::nn::Linear linear;
    public:
        /**
         * @brief Constructor for CategoricalOutput layer.
         *
         * @param numInputs Number of input features from the previous network layer.
         * @param numOutputs Number of discrete actions available to the agent.
         */
        CategoricalOutput(unsigned int numInputs, unsigned int numOutputs);

        /**
         * @brief Computes Categorical distribution from input tensor.
         *
         * @param x Input tensor of shape (batch_size, num_inputs) containing network features.
         * @return std::unique_ptr<Distribution> A Categorical distribution that encodes the
         *         probability of taking each action for each sample in the batch.
         */
        std::unique_ptr<Distribution> forward(torch::Tensor x) override;
    };

    /**
         * @class NormalOutput
         * @brief Output layer that produces Normal (Gaussian) distributions for continuous action spaces.
         *
         * NormalOutput is used for continuous control problems where actions are real-valued vectors
         * (e.g., robot joint angles, steering angles). It transforms input features into the mean
         * (location) and log standard deviation parameters of a Normal distribution, enabling the
         * agent to learn both the action mean and its uncertainty.
         *
         * The layer uses two separate linear transformations: one for the mean of each action dimension,
         * and one shared log-variance parameter that controls the exploration/exploitation trade-off.
    */
    class NormalOutput : public OutputLayer
    {
    private:
        /**
          * @brief Linear layer for computing the mean (location) of the Normal distribution.
          *
          * Maps from num_inputs features to num_outputs values representing the mean action
          * for each continuous action dimension.
          */
        torch::nn::Linear linear_loc;

        /**
         * @brief Log standard deviation parameter of the Normal distribution.
         *
         * A learnable parameter that represents log(σ) where σ is the standard deviation.
         * Using log-space ensures the standard deviation remains positive. This is typically
         * shared across the batch to provide consistent exploration magnitude.
         */
        torch::Tensor scale_log;

    public:
        /**
    * @brief Constructor for NormalOutput layer.
    *
    * @param num_inputs Number of input features from the previous network layer.
    * @param num_outputs Dimensionality of the continuous action space.
    */
        NormalOutput(unsigned int num_inputs, unsigned int num_outputs);

        /**
         * @brief Computes Normal distribution from input tensor.
         *
         * @param x Input tensor of shape (batch_size, num_inputs) containing network features.
         * @return std::unique_ptr<Distribution> A Normal distribution parameterized by the
         *         computed mean (from linear_loc) and log standard deviation (scale_log).
         *         The returned distribution can sample continuous actions and compute log-probabilities.
         */
        std::unique_ptr<Distribution> forward(torch::Tensor x) override;

    };
}



#endif //LUNARALIGHTINGRL_OUTPUTLAYERS_HPP