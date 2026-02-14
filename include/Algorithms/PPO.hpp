#pragma once
//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_PPO_HPP
#define LUNARALIGHTINGRL_PPO_HPP
#include<string>
#include<vector>



#include"Algorithm.hpp"
#include<torch/torch.h>


namespace LunarAlighting
{
    class Policy;
    class RolloutStorge;
    /**
     * @class PPO
     * @brief Proximal Policy Optimization (PPO) algorithm implementation.
     *
     * PPO is a state-of-the-art policy gradient algorithm that addresses training instability in
     * earlier methods through a clipping mechanism. Instead of directly maximizing the policy gradient,
     * PPO clips the probability ratio between the new and old policies to prevent excessively large
     * policy updates. This ensures more stable and predictable training compared to unconstrained
     * policy gradient methods.
     *
     * @details
     * Key distinguishing features from A2C:
     * - **Clipped objective**: Policy updates are constrained using min(ratio, clipped_ratio)
     * - **Multiple epochs**: Rollout data is reused across multiple training epochs for efficiency
     * - **Mini-batching**: Rollouts are divided into mini-batches for stochastic gradient descent
     * - **KL divergence monitoring**: Adaptive clipping based on KL divergence between old and new policies
     * - **Adam optimizer**: Uses adaptive learning rates for improved convergence
     *
     * The clipping mechanism balances two competing objectives:
     * 1. Maximize expected return using policy gradients
     * 2. Maintain policy similarity to avoid destructive updates
     *
     * @see Algorithm - Base class defining the update interface
     */
    class PPO : public Algorithms
    {
    private:
        Policy &policy;                                    ///< Reference to the neural network policy (actor-critic network)
        float actorLossCoef;                             ///< Coefficient scaling the clipped actor (policy) loss component
        float valueLossCoef;                             ///< Coefficient scaling the value (critic) loss component
        float entropyCoef;                                ///< Coefficient for entropy regularization to maintain exploration
        float maxGradNorm;                               ///< Maximum L2 norm threshold for gradient clipping to ensure training stability
        float originalLearningRate;                      ///< Initial learning rate; used for scheduling and restoration
        float originalClipParam;                         ///< Initial clipping parameter (epsilon in PPO paper); controls policy constraint radius
        float kl_target;                                   ///< Target KL divergence for adaptive clipping; used to adjust clip_param dynamically
        int numEpoch;                                     ///< Number of gradient descent epochs per rollout; reuses collected data for multiple updates
        int numMiniBatch;                                ///< Number of mini-batches for stochastic gradient descent; divides rollout data for efficiency
        std::unique_ptr<torch::optim::Adam> optimizer;     ///< Adam optimizer managing parameter updates with adaptive learning rates per parameter
    public:
         /**
         * @brief Constructs a PPO algorithm instance with specified hyperparameters.
         *
         * @param policy Reference to the neural network policy implementing both actor and critic heads.
         *               The policy must output both action probabilities (actor) and state values (critic).
         *
         * @param clip_param Clipping range for the probability ratio in the objective function.
         *                   Typical range: [0.1, 0.3]. Controls maximum allowed policy update magnitude.
         *                   PPO clips the ratio to [1 - clip_param, 1 + clip_param].
         *                   Larger values permit more aggressive updates; smaller values enforce conservative changes.
         *
         * @param num_epoch Number of times the collected rollouts are reused for training. Typical range: [3, 10].
         *                  Higher values maximize data efficiency but risk overfitting to stale data.
         *                  Must be ≥ 1. Each epoch performs a full pass through the rollout data.
         *
         * @param num_mini_batch Number of mini-batches sampled from rollouts during each epoch. Typical range: [4, 32].
         *                        Controls batch size: actual_batch_size = total_steps / num_mini_batch.
         *                        More mini-batches provide noisier gradient estimates; fewer batches use more memory.
         *
         * @param actor_loss_coef Multiplier for the clipped policy gradient loss. Typical range: [0.5, 2.0].
         *                        Higher values emphasize policy improvement over value prediction.
         *
         * @param value_loss_coef Multiplier for the value function loss. Typical range: [0.5, 2.0].
         *                        Controls importance of accurate value estimation for advantage computation.
         *
         * @param entropy_coef Coefficient for entropy regularization. Typical range: [0.001, 0.1].
         *                     Encourages exploration; often annealed during training from high to low values.
         *
         * @param learning_rate Step size for Adam optimizer. Typical range: [1e-5, 1e-4].
         *                      Usually smaller than A2C due to multiple epochs reusing same data.
         *
         * @param epsilon Small constant for numerical stability in Adam. Default: 1e-8.
         *                Prevents division by zero in adaptive learning rate computation.
         *
         * @param max_grad_norm Maximum L2 norm for gradient clipping. Default: 0.5.
         *                      Prevents exploding gradients during backpropagation.
         *
         * @param kl_target Target KL divergence for adaptive clipping. Default: 0.01.
         *                  If actual KL > 2 * kl_target, clip_param decreases (conservative).
         *                  If actual KL < 0.5 * kl_target, clip_param increases (aggressive).
         *                  Enables automatic adjustment of policy constraint during training.
         */
        PPO(Policy &policy,
            float clipParam,
            int numEpoch,
            int numMiniBatch,
            float actorLossCoef,
            float valueLossCoef,
            float entropyCoef,
            float learningRate,
            float epsilon = 1e-8,
            float maxGradNorm = 0.5,
            float kl_target = 0.01);

        /**
         * @brief Performs one complete optimization step using collected rollouts.
         *
         * Implements the PPO update procedure:
         * 1. Divides rollout data into num_mini_batch mini-batches
         * 2. For each of num_epoch epochs:
         *    - Iterates through all mini-batches
         *    - Computes clipped policy gradient, value, and entropy losses
         *    - Backpropagates gradients and applies Adam updates
         * 3. Applies gradient clipping to prevent exploding gradients
         * 4. Optionally adjusts clip_param based on KL divergence feedback
         *
         * @param rollouts Reference to RolloutStorage containing:
         *                 - Collected observations, actions, and rewards from environment interactions
         *                 - Computed advantages and returns for value target computation
         *                 - Old policy log-probabilities needed for ratio computation
         *
         * @param decay_level Learning rate decay multiplier in [0, 1]. Default: 1 (no decay).
         *                    Value < 1 reduces effective learning rate: lr_effective = lr * decay_level.
         *                    Enables progressive learning rate scheduling across training phases.
         *
         * @return std::vector<UpdateDatum> Statistics from the update containing:
         *         - Policy loss value indicating gradient alignment and clipping effectiveness
         *         - Value loss value indicating baseline accuracy
         *         - Entropy loss value indicating exploration level
         *         - KL divergence between old and new policies (diagnostic for clip_param adjustment)
         *         - Clipping fraction (percentage of probability ratios that hit clip bounds)
         *         - Other diagnostic metrics for monitoring training progress and stability
         *
         * @note Modifies optimizer state; processes entire rollout dataset across epochs and mini-batches.
         * @note The multiple-epoch structure means gradient information from the same trajectory is used repeatedly,
         *       which requires lower learning rates compared to single-epoch algorithms like A2C.
         * @see RolloutStorage - Data structure organizing collected experience for gradient computation
         */
        std::vector<UpdateDatum> update(RolloutStorge &rollouts, float decay_level = 1) override;
    };
}

#endif //LUNARALIGHTINGRL_PPO_HPP