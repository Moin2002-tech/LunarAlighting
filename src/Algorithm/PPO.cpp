/**
 * @file PPO.cpp
 * @brief Implementation of Proximal Policy Optimization (PPO) algorithm.
 * @author moinshaikh
 * @date 2/8/26
 * 
 * This file contains the complete implementation of the PPO algorithm, including:
 * - Constructor initialization with hyperparameters
 * - Core update() method implementing PPO's clipped objective
 * - Test utilities for validating learning behavior
 * - Comprehensive test cases for different scenarios
 * 
 * PPO is a state-of-the-art policy gradient method that improves training stability
 * by clipping the policy update to prevent excessively large changes.
 */

#include<chrono>
#include<memory>

#include<torch/torch.h>
#include"../../include/Algorithms/PPO.hpp"
#include"../../include/Algorithms/Algorithm.hpp"
#include"../../include/Generator/Generator.hpp"
#include"../../include/Model/mlp_base.hpp"
#include"../../include/Model/policy.hpp"
#include"../../include/Storage.hpp"
#include"../../include/Space.hpp"
#include "../../include/Algorithms/A2c.hpp"
#include"../third_party/doctest.hpp"

namespace LunarAlighting {

/**
 * @brief PPO constructor - initializes algorithm with hyperparameters.
 * 
 * This constructor sets up all the necessary components for PPO training:
 * - Stores hyperparameters for the clipped objective function
 * - Initializes Adam optimizer with adaptive learning rates
 * - Configures clipping parameters for policy stability
 * 
 * @param policy Reference to the neural network policy (actor-critic architecture)
 * @param clipParam Clipping range epsilon (typically 0.1-0.3). Constrains policy updates.
 *                 The probability ratio r = π_new/π_old is clipped to [1-ε, 1+ε].
 * @param numEpoch Number of training epochs per rollout (typically 3-10).
 *                 Higher values reuse data more efficiently but risk overfitting.
 * @param numMiniBatch Number of mini-batches for stochastic gradient descent.
 *                    Controls batch size: actual_batch = total_steps / numMiniBatch.
 * @param actorLossCoef Weight for policy gradient loss (typically 0.5-2.0).
 *                    Balances policy improvement vs value prediction.
 * @param valueLossCoef Weight for value function loss (typically 0.5-2.0).
 *                    Controls importance of accurate value estimates.
 * @param entropyCoef Weight for entropy regularization (typically 0.001-0.1).
 *                   Encourages exploration; often annealed during training.
 * @param learningRate Adam optimizer step size (typically 1e-5 to 1e-4).
 *                    Smaller than A2C due to multiple epochs on same data.
 * @param epsilon Adam's numerical stability constant (default: 1e-8).
 *                Prevents division by zero in adaptive learning rate computation.
 * @param maxGradNorm Maximum gradient L2 norm for clipping (default: 0.5).
 *                   Prevents exploding gradients during backpropagation.
 * @param kl_target Target KL divergence for adaptive clipping (default: 0.01).
 *                 Triggers early stopping if KL > 1.5 * kl_target.
 */
    PPO::PPO(Policy &policy,
        float clipParam,
        int numEpoch,
        int numMiniBatch,
        float actorLossCoef,
        float valueLossCoef,
        float entropyCoef,
        float learningRate,
        float epsilon,
        float maxGradNorm,
        float kl_target) :
    policy(policy),
    actorLossCoef(actorLossCoef),
    valueLossCoef(valueLossCoef),
    entropyCoef(entropyCoef),
    maxGradNorm(maxGradNorm),
    originalClipParam(clipParam),
    originalLearningRate(learningRate),
    kl_target(kl_target),
    numEpoch(numEpoch),
    numMiniBatch(numMiniBatch),
    optimizer(std::make_unique<torch::optim::Adam> (policy->parameters(),torch::optim::AdamOptions(learningRate).eps(epsilon)))
    {
        // Constructor body - all initialization done via member initializer list
        // Adam optimizer automatically handles adaptive learning rates per parameter
        // epsilon parameter ensures numerical stability in denominator
    }

/**
 * @brief Core PPO update method - performs clipped policy optimization.
 * 
 * This method implements the complete PPO algorithm:
 * 1. **Advantage Computation**: Calculates A_t = R_t - V(s_t) for each timestep
 * 2. **Advantage Normalization**: Standardizes advantages to zero mean, unit variance
 * 3. **Multi-Epoch Training**: Reuses rollout data across multiple epochs
 * 4. **Mini-Batch Processing**: Divides data into batches for SGD
 * 5. **Clipped Objective**: Applies PPO's clipping mechanism to prevent large policy updates
 * 6. **Early Stopping**: Monitors KL divergence to avoid destructive updates
 * 
 * **Key PPO Components:**
 * - **Probability Ratio**: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
 * - **Clipped Surrogate**: L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
 * - **Value Loss**: L^VF(θ) = (V_θ(s_t) - V_t^target)²
 * - **Entropy Bonus**: L^ent(θ) = -Σ π_θ(a|s) log π_θ(a|s)
 * 
 * **Advantage Normalization:**
 * Normalizing advantages (subtract mean, divide by std + 1e-5) stabilizes training
 * by ensuring consistent scale across updates and reducing variance.
 * 
 * **KL Early Stopping:**
 * If KL divergence exceeds 1.5 × kl_target, training stops early to prevent
 * policy collapse. This is a safety mechanism against destructive updates.
 * 
 * @param rolloutStorage Container with collected experience (observations, actions, rewards)
 * @param decayLevel Learning rate decay multiplier in [0, 1]. Default: 1 (no decay).
 *                   Allows progressive learning rate scheduling across training.
 * 
 * @return Vector of UpdateDatum containing training statistics:
 *         - "Value loss": MSE between predicted and target values
 *         - "Action loss": Clipped policy gradient loss
 *         - "Clip fraction": Percentage of ratios hitting clip bounds
 *         - "Entropy": Policy entropy (exploration measure)
 *         - "KL divergence": Average KL between old and new policies
 *         - "KL divergence early stop update": Update count when early stopped (if applicable)
 */
    std::vector<UpdateDatum> PPO::update(RolloutStorge &rolloutStorage,float decayLevel) {
        // Apply learning rate decay to both clipping parameter and optimizer
        float clipParams = originalClipParam * decayLevel;

        // Update Adam optimizer learning rate for all parameter groups
        for (auto& group :optimizer->param_groups())
        {
            static_cast<torch::optim::AdamOptions&> (group.options()).lr(originalLearningRate* decayLevel);
        }

        // ===========================================
        // ADVANTAGE COMPUTATION
        // ===========================================
        // Advantages measure how much better an action was than expected
        // A_t = R_t - V(s_t) where R_t is discounted return, V(s_t) is value prediction
        auto returns = rolloutStorage.get_returns();
        auto value_preds = rolloutStorage.get_value_predictions();
        auto advantages = (returns.narrow(0, 0, returns.size(0) - 1) -
                           value_preds.narrow(0, 0, value_preds.size(0) - 1));

        // ===========================================
        // ADVANTAGE NORMALIZATION
        // ===========================================
        // Standardize advantages: (A - μ) / (σ + ε)
        // This stabilizes training by ensuring consistent scale and reducing variance
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5);

        // Initialize training statistics tracking
        float total_value_loss = 0;      // Accumulated value function MSE
        float total_action_loss = 0;      // Accumulated clipped policy loss
        float total_entropy = 0;          // Accumulated policy entropy
        float kl_divergence = 0;          // KL between old and new policies
        float kl_early_stopped = -1;      // Update count when early stopped (-1 = none)
        float clip_fraction = 0;          // Percentage of ratios hitting clip bounds
        int num_updates = 0;              // Total number of gradient updates performed

        // ===========================================
        // MULTI-EPOCH TRAINING LOOP
        // ===========================================
        // PPO reuses the same rollout data for multiple epochs to improve sample efficiency
        // Each epoch processes the entire dataset with different mini-batch orderings
        for (int epoch = 0; epoch < numEpoch; ++epoch)
        {
            // ===========================================
            // DATA GENERATOR SETUP
            // ===========================================
            // Create appropriate generator based on policy architecture
            std::unique_ptr<Generator> data_generator;
            if (policy->is_recurrent())
            {
                // For recurrent policies: maintain sequence continuity across batches
                data_generator = rolloutStorage.recurrentGenerator(advantages,
                                                              numMiniBatch);
            }
            else
            {
                // For feed-forward policies: can shuffle sequences freely
                data_generator = rolloutStorage.feed_forward_generator(advantages,
                                                                 numMiniBatch);
            }

            // ===========================================
            // MINI-BATCH PROCESSING LOOP
            // ===========================================
            // Process shuffled rollout data in mini-batches for stochastic gradient descent
            while (!data_generator->done()) {
                MiniBatch mini_batch = data_generator->next();

                // ===========================================
                // POLICY EVALUATION
                // ===========================================
                // Forward pass through policy network to get:
                // evaluate_result[0] = value predictions V(s)
                // evaluate_result[1] = action log probabilities log π(a|s)
                // evaluate_result[2] = policy entropy H(π)
                auto evaluate_result = policy->evaluateAction(
                    mini_batch.observations,
                    mini_batch.hiddenStates,
                    mini_batch.masks,
                    mini_batch.actions);

                // ===========================================
                // KL DIVERGENCE MONITORING
                // ===========================================
                // KL divergence measures how much the policy has changed:
                // KL(π_old || π_new) = Σ π_old(a|s) log(π_old(a|s) / π_new(a|s))
                // Used for early stopping to prevent destructive policy updates
                kl_divergence = (mini_batch.actionLogProbs - evaluate_result[1])
                                    .mean()
                                    .item()
                                    .toFloat();
                
                // Early stopping: if policy changes too much, stop training
                if (kl_divergence > kl_target * 1.5)
                {
                    kl_early_stopped = num_updates;
                    goto finish_update;  // Exit both loops safely
                }

                // ===========================================
                // PROBABILITY RATIO COMPUTATION
                // ===========================================
                // r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log π_new - log π_old)
                // This ratio measures how much the policy has changed for each action
                auto ratio = torch::exp(evaluate_result[1] -
                                        mini_batch.actionLogProbs);

                // ===========================================
                // PPO CLIPPED OBJECTIVE FUNCTION
                // ===========================================
                // L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
                // 
                // Two scenarios:
                // 1. If A_t > 0 (good action): maximize r_t but clip at 1+ε
                // 2. If A_t < 0 (bad action): minimize r_t but clip at 1-ε
                // 
                // This prevents large policy updates that could destroy performance
                auto surr_1 = ratio * mini_batch.advantages;  // Unclipped objective
                auto surr_2 = (torch::clamp(ratio,
                                            1.0 - clipParams,
                                            1.0 + clipParams) *
                               mini_batch.advantages);  // Clipped objective
                
                // Track how often ratios hit clipping bounds (diagnostic)
                clip_fraction += (ratio - 1.0)
                                     .abs()
                                     .gt(clipParams)
                                     .to(torch::kFloat)
                                     .mean()
                                     .item()
                                     .toFloat();
                
                // Take minimum of clipped and unclipped (pessimistic bound)
                auto action_loss = -torch::min(surr_1, surr_2).mean();

                // ===========================================
                // VALUE FUNCTION LOSS
                // ===========================================
                // L^VF(θ) = 0.5 * (V_θ(s_t) - V_t^target)²
                // Trains the critic to accurately predict state values
                // The 0.5 factor cancels with the derivative of squared error
                auto value_loss = 0.5 * (mini_batch.returns - evaluate_result[0])
                                            .pow(2)
                                            .mean();
                // TODO: Implement clipped value loss (PPO variant with value clipping)
                // This would provide additional stability for value function learning

                // ===========================================
                // TOTAL LOSS COMPOSITION
                // ===========================================
                // L(θ) = L^CLIP(θ) - c1 * L^VF(θ) + c2 * L^ent(θ)
                // 
                // Components:
                // - L^CLIP: Clipped policy objective (maximize)
                // - L^VF: Value function loss (minimize)
                // - L^ent: Entropy bonus (maximize for exploration)
                // 
                // Note signs: action_loss is negative (maximization), entropy is subtracted (maximization)
                auto loss = (value_loss * valueLossCoef +
                             action_loss * actorLossCoef -
                             evaluate_result[2] * entropyCoef);

                // ===========================================
                // GRADIENT OPTIMIZATION STEP
                // ===========================================
                optimizer->zero_grad();  // Clear previous gradients
                loss.backward();         // Compute gradients via backpropagation
                // TODO: Implement gradient norm clipping to prevent exploding gradients
                // torch::nn::utils::clip_grad_norm_(policy->parameters(), maxGradNorm);
                optimizer->step();        // Update parameters using Adam optimizer
                num_updates++;           // Track total number of updates

                // Accumulate statistics for reporting
                total_value_loss += value_loss.item().toFloat();
                total_action_loss += action_loss.item().toFloat();
                total_entropy += evaluate_result[2].item().toFloat();
            }
        }
        
        finish_update:
        // ===========================================
        // OBSERVATION NORMALIZATION UPDATE
        // ===========================================
        // If policy uses observation normalization, update running statistics
        // This normalizes inputs to zero mean, unit variance for stable training
    if (policy->usingObservationNormalizer())
    {
        policy->updateObervationNormalizer(rolloutStorage.get_observations());
    }

        // ===========================================
        // STATISTICS NORMALIZATION
        // ===========================================
        // Average accumulated statistics over number of updates
        total_value_loss /= num_updates;
        total_action_loss /= num_updates;
        total_entropy /= num_updates;
        clip_fraction /= num_updates;

        // ===========================================
        // RETURN TRAINING STATISTICS
        // ===========================================
        if (kl_early_stopped > -1)
        {
            // Early stopping occurred - include early stop info
            return {{"Value loss", total_value_loss},
                    {"Action loss", total_action_loss},
                    {"Clip fraction", clip_fraction},
                    {"Entropy", total_entropy},
                    {"KL divergence", kl_divergence},
                    {"KL divergence early stop update", kl_early_stopped}};
        }
        else
        {
            // Normal completion - no early stopping
            return {{"Value loss", total_value_loss},
                    {"Action loss", total_action_loss},
                    {"Clip fraction", clip_fraction},
                    {"Entropy", total_entropy},
                    {"KL divergence", kl_divergence}};
        }
    }


    /**
 * @brief Test utility: trains policy to learn a simple action-reward pattern.
 * 
 * This function creates a simple learning scenario where the reward equals the action.
 * The policy should learn to prefer action 1 over action 0 since:
 * - Action 0 → reward 0
 * - Action 1 → reward 1
 * 
 * **Training Process:**
 * 1. Generate random binary observations (0 or 1)
 * 2. Policy selects actions based on current probabilities
 * 3. Rewards equal the selected actions
 * 4. PPO updates policy to maximize expected reward
 * 5. Repeat for 10 episodes × 5 timesteps each
 * 
 * **Expected Learning:**
 * After training, policy should increase probability of action 1
 * and decrease probability of action 0.
 * 
 * @param policy Reference to the policy network being trained
 * @param storage Rollout storage for collecting experience
 * @param ppo PPO algorithm instance for performing updates
 */
static void learn_pattern(Policy &policy, RolloutStorge &storage, PPO &ppo)
{
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            auto observation = torch::randint(0, 2, {2, 1}).to(torch::kFloat);

            std::vector<torch::Tensor> act_result;
            {
                torch::NoGradGuard no_grad;
                act_result = policy->act(observation,
                                         torch::Tensor(),
                                         torch::ones({2, 1}));
            }
            auto actions = act_result[1];

            auto rewards = actions;
            storage.insert(observation,
                           torch::zeros({2, 5}),
                           actions,
                           act_result[2],
                           act_result[0],
                           rewards,
                           torch::ones({2, 1}));
        }

        torch::Tensor next_value;
        {
            torch::NoGradGuard no_grad;
            next_value = policy->getValue(
                                   storage.get_observations()[-1],
                                   storage.get_hidden_states()[-1],
                                   storage.get_masks()[-1])
                             .detach();
        }
        storage.computeReturns(next_value, false, 0., 0.9);

        ppo.update(storage);
        storage.afterUpdate();
    }
}

/**
 * @brief Test utility: trains policy to learn a simple matching game.
 * 
 * This function implements a simple game where:
 * - If action matches observation → reward +1
 * - If action doesn't match observation → reward -1
 * 
 * **Game Logic:**
 * The policy must learn to copy the observation to maximize reward.
 * This tests the policy's ability to learn input-output mappings.
 * 
 * **Training Process:**
 * 1. Generate random binary observation (0 or 1)
 * 2. Policy selects action based on observation
 * 3. Reward = +1 if action == observation, else -1
 * 4. PPO updates policy to maximize expected reward
 * 5. Repeat for 10 episodes × 5 timesteps each
 * 
 * **Expected Learning:**
 * After training, policy should learn to:
 * - When observation = 0 → prefer action 0
 * - When observation = 1 → prefer action 1
 * 
 * @param policy Reference to the policy network being trained
 * @param storage Rollout storage for collecting experience
 * @param ppo PPO algorithm instance for performing updates
 */
static void learn_game(Policy &policy, RolloutStorge &storage, PPO &ppo)
{
    // The game is: If the action matches the input, give a reward of 1, otherwise -1
    auto observation = torch::randint(0, 2, {2, 1}).to(torch::kFloat);
    storage.setFirstObservation(observation);

    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            std::vector<torch::Tensor> act_result;
            {
                torch::NoGradGuard no_grad;
                act_result = policy->act(observation,
                                         torch::Tensor(),
                                         torch::ones({2, 1}));
            }
            auto actions = act_result[1];

            auto rewards = ((actions.to(torch::kLong) == observation.to(torch::kLong)).to(torch::kFloat) * 2) - 1;
            observation = torch::randint(0, 2, {2, 1}).to(torch::kFloat);
            storage.insert(observation,
                           torch::zeros({2, 5}),
                           actions,
                           act_result[2],
                           act_result[0],
                           rewards,
                           torch::ones({2, 1}));
        }

        torch::Tensor next_value;
        {
            torch::NoGradGuard no_grad;
            next_value = policy->getValue(
                                   storage.get_observations()[-1],
                                   storage.get_hidden_states()[-1],
                                   storage.get_masks()[-1])
                             .detach();
        }
        storage.computeReturns(next_value, false, 0.1, 0.9);

        ppo.update(storage);
        storage.afterUpdate();
    }
}

/**
 * @brief Comprehensive test suite for PPO algorithm implementation.
 * 
 * This test suite validates that PPO can learn simple tasks and that
 * the policy probabilities change in the expected directions after training.
 * 
 * **Test Structure:**
 * 1. **Pattern Learning Test**: Validates learning of action-reward association
 * 2. **Game Learning Tests**: Validates learning of input-output mappings
 *    - Without observation normalization
 *    - With observation normalization
 * 
 * **Test Validation:**
 * Each test measures policy probabilities before and after training,
 * then verifies they changed in the expected direction:
 * - Probability of suboptimal action should decrease
 * - Probability of optimal action should increase
 * 
 * **Why These Tests Matter:**
 * - Pattern learning tests basic reward maximization
 * - Game learning tests conditional behavior (observation → action)
 * - Normalization tests robustness to input preprocessing
 * 
 * The tests use torch::manual_seed(0) to ensure reproducible results.
 */
TEST_CASE("PPO")
{
    torch::manual_seed(0);
    /**
     * @brief Test: PPO learns basic action-reward pattern.
     * 
     * This subcase validates that PPO can learn a simple pattern where
     * the reward equals the action taken. The policy should learn to
     * prefer action 1 (reward 1) over action 0 (reward 0).
     * 
     * **Test Setup:**
     * - MLP base network with 1 input, 5 hidden units
     * - Discrete action space with 2 actions
     * - No observation normalization
     * - 10 training episodes
     * 
     * **Success Criteria:**
     * - P(action=0) should decrease after training
     * - P(action=1) should increase after training
     */
    SUBCASE("update() learns basic pattern")
    {
        auto base = std::make_shared<MlpBase>(1, false, 5);
        ActionSpace space{"Discrete", {2}};
        Policy policy(space, base, false);
        RolloutStorge storage(20, 2, {1}, space, 5, torch::kCPU);
        PPO ppo(policy, 0.2, 3, 5, 1, 0.5, 1e-3, 0.001);

        // The reward is the action
        auto pre_game_probs = policy->getProbability(
            torch::ones({2, 1}),
            torch::zeros({2, 5}),
            torch::ones({2, 1}));

        learn_pattern(policy, storage, ppo);

        auto post_game_probs = policy->getProbability(
            torch::ones({2, 1}),
            torch::zeros({2, 5}),
            torch::ones({2, 1}));

        INFO("Pre-training probabilities: \n"
             << pre_game_probs << "\n");
        INFO("Post-training probabilities: \n"
             << post_game_probs << "\n");
        CHECK(post_game_probs[0][0].item().toDouble() <
              pre_game_probs[0][0].item().toDouble());
        CHECK(post_game_probs[0][1].item().toDouble() >
              pre_game_probs[0][1].item().toDouble());
    }

    /**
     * @brief Test: PPO learns basic conditional game.
     * 
     * This subcase validates that PPO can learn a conditional mapping
     * where the optimal action depends on the observation. The policy
     * should learn to copy the observation to maximize reward.
     * 
     * **Game Rules:**
     * - If action == observation → reward +1
     * - If action != observation → reward -1
     * 
     * **Test Variants:**
     * 1. Without observation normalization: tests basic learning
     * 2. With observation normalization: tests robustness to preprocessing
     * 
     * **Success Criteria:**
     * For observation = 1:
     * - P(action=0) should decrease after training
     * - P(action=1) should increase after training
     */
    SUBCASE("update() learns basic game")
    {
        /**
         * @brief Test game learning without observation normalization.
         * 
         * This variant tests basic conditional learning where the policy
         * receives raw binary observations and must learn the matching rule.
         */
        SUBCASE("Without observation normalization")
        {
            auto base = std::make_shared<MlpBase>(1, false, 5);
            ActionSpace space{"Discrete", {2}};
            Policy policy(space, base, false);
            RolloutStorge storage(20, 2, {1}, space, 5, torch::kCPU);
            PPO ppo(policy, 0.2, 3, 5, 1, 0.5, 1e-3, 0.001);

            // The game is: If the action matches the input, give a reward of 1, otherwise -1
            auto pre_game_probs = policy->getProbability(
                torch::ones({2, 1}),
                torch::zeros({2, 5}),
                torch::ones({2, 1}));

            learn_game(policy, storage, ppo);

            auto post_game_probs = policy->getProbability(
                torch::ones({2, 1}),
                torch::zeros({2, 5}),
                torch::ones({2, 1}));

            INFO("Pre-training probabilities: \n"
                 << pre_game_probs << "\n");
            INFO("Post-training probabilities: \n"
                 << post_game_probs << "\n");
            CHECK(post_game_probs[0][0].item().toDouble() <
                  pre_game_probs[0][0].item().toDouble());
            CHECK(post_game_probs[0][1].item().toDouble() >
                  pre_game_probs[0][1].item().toDouble());
        }

        /**
         * @brief Test game learning with observation normalization.
         * 
         * This variant tests conditional learning when observations are
         * normalized to zero mean and unit variance. The policy should
         * still learn the matching rule despite the preprocessing.
         * 
         * **Why This Test Matters:**
         * Observation normalization is common in deep RL for stability.
         * This test ensures PPO works correctly with normalized inputs.
         */
        SUBCASE("With observation normalization")
        {
            auto base = std::make_shared<MlpBase>(1, false, 5);
            ActionSpace space{"Discrete", {2}};
            Policy policy(space, base, true);
            RolloutStorge storage(20, 2, {1}, space, 5, torch::kCPU);
            PPO ppo(policy, 0.2, 3, 5, 1, 0.5, 1e-3, 0.001);

            // The game is: If the action matches the input, give a reward of 1, otherwise -1
            auto pre_game_probs = policy->getProbability(
                torch::ones({2, 1}),
                torch::zeros({2, 5}),
                torch::ones({2, 1}));

            learn_game(policy, storage, ppo);

            auto post_game_probs = policy->getProbability(
                torch::ones({2, 1}),
                torch::zeros({2, 5}),
                torch::ones({2, 1}));

            INFO("Pre-training probabilities: \n"
                 << pre_game_probs << "\n");
            INFO("Post-training probabilities: \n"
                 << post_game_probs << "\n");
            CHECK(post_game_probs[0][0].item().toDouble() <
                  pre_game_probs[0][0].item().toDouble());
            CHECK(post_game_probs[0][1].item().toDouble() >
                  pre_game_probs[0][1].item().toDouble());
        }
    }
}

}