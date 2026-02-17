#pragma once

//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_A2C_HPP

#define LUNARALIGHTINGRL_A2C_HPP

#include"Algorithm.hpp"
#include<torch/torch.h>


namespace LunarAlighting {
    class Policy;
    class RolloutStorge;

    /**
     * @class A2C
     * @brief Advantage Actor-Critic (A2C) algorithm implementation.
     *
     * A2C is a synchronous variant of the Asynchronous Advantage Actor-Critic (A3C) algorithm.
     * It combines policy gradient methods (actor) with value function approximation (critic) to
     * learn both action selection and state value estimation concurrently. The algorithm leverages
     * the advantage function (reward - baseline) to reduce variance in gradient estimates while
     * maintaining unbiased policy gradients.
     *
     * @details
     * This implementation uses PyTorch's RMSprop optimizer for updating network parameters.
     * The algorithm balances three loss components:
     * - Actor loss: Policy gradient loss weighted by advantages
     * - Value loss: Temporal difference loss from value function predictions
     * - Entropy regularization: Encourages exploration during early training phases
     *
     * @see Algorithm - Base class defining the update interface
    */
    class A2C : public Algorithms
    {
    private:
        Policy &policy;                                    ///< Reference to the neural network policy (actor-critic network)
        float actorLossCoef;                             ///< Coefficient scaling the actor (policy) loss component
        float valueLossCoef;                             ///< Coefficient scaling the value (critic) loss component
        float entropyCoef;                                ///< Coefficient for entropy regularization to promote exploration
        float maxGradNorm;                               ///< Maximum norm threshold for gradient clipping to ensure training stability
        float originalLearningRate;                      ///< Initial learning rate; used for scheduling and restoration
        std::unique_ptr<torch::optim::RMSprop> optimizer;  ///< RMSprop optimizer managing parameter updates during training
    public:
        /**
         * @brief Constructs an A2C algorithm instance with specified hyperparameters.
         *
         * @param policy Reference to the neural network policy implementing both actor and critic heads.
         *               The policy must output both action probabilities (actor) and state values (critic).
         *
         * @param actor_loss_coef Multiplier for the policy gradient loss. Typical range: [0.5, 2.0].
         *                        Higher values emphasize policy improvement over value prediction.
         *
         * @param value_loss_coef Multiplier for the value function loss. Typical range: [0.5, 2.0].
         *                        Controls importance of accurate value estimation for advantage computation.
         *
         * @param entropy_coef Coefficient for entropy regularization. Typical range: [0.001, 0.1].
         *                     Encourages exploration; can be annealed during training.
         *
         * @param learning_rate Step size for gradient descent. Typical range: [1e-4, 1e-3].
         *                      Controls convergence speed and stability.
         *
         * @param epsilon Small constant for numerical stability in RMSprop. Default: 1e-8.
         *                Prevents division by zero in adaptive learning rate computation.
         *
         * @param alpha Decay rate for RMSprop's exponential moving average. Default: 0.99.
         *              Controls the influence of historical gradient statistics.
         *
         * @param max_grad_norm Maximum L2 norm for gradient clipping. Default: 0.5.
         *                      Prevents exploding gradients during training.
         */
        A2C(Policy &policy,
        float actorLossCoef,
        float valueLossCoef,
        float entropyCoef,
        float learningRate,
        float epsilon = 1e-8,
        float alpha = 0.99,
        float maxGradNorm = 0.5);
        /**
         * @brief Performs one optimization step using collected rollouts.
         *
         * Computes policy and value function losses from the provided rollout data, backpropagates
         * gradients, applies gradient clipping, and updates network parameters via RMSprop.
         *
         * @param rollouts Reference to RolloutStorage containing:
         *                 - Collected observations, actions, and rewards from environment interactions
         *                 - Computed advantages from value function baseline
         *                 - Returns (cumulative discounted rewards) for value target computation
         *
         * @param decay_level Learning rate decay multiplier in [0, 1]. Default: 1 (no decay).
         *                    Value < 1 reduces effective learning rate: lr_effective = lr * decay_level.
         *                    Enables progressive learning rate scheduling across training phases.
         *
         * @return std::vector<UpdateDatum> Statistics from the update containing:
         *         - Actor loss value indicating policy improvement quality
         *         - Value loss value indicating baseline accuracy
         *         - Entropy loss value indicating exploration level
         *         - Other diagnostic metrics for monitoring training progress
         *
         * @note Modifies optimizer state; should be called once per training iteration.
         * @see RolloutStorage - Data structure organizing collected experience for gradient computation
         */
        std::vector<UpdateDatum> update(RolloutStorge &rollouts, float decay_level = 1);
    };
};






#endif //LUNARALIGHTINGRL_A2C_HPP