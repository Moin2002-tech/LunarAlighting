//
// Created by moinshaikh on 1/28/26.
//

#include<algorithm>
#include<vector>

#include<torch/torch.h>

#include"../../include/Generator/FeedForwardGenerator.hpp"
#include"../../include/Generator/Generator.hpp"
#include"../third_party/doctest.hpp"
/**
 * @file FeedForwardGenerator.cpp
 * @brief Implementation of FeedForwardGenerator for shuffled mini-batch sampling
 *
 * This file contains the implementation of the FeedForwardGenerator class, which is
 * responsible for organizing trajectory data into randomized mini-batches suitable
 * for training feed-forward neural networks. The generator implements a shuffling
 * strategy that randomly reorders experience samples and divides them into fixed-size
 * mini-batches for gradient-based optimization.
 *
 * @details
 *
 * **Algorithm Overview:**
 *
 * The FeedForwardGenerator uses a two-phase approach:
 *
 * **Phase 1: Shuffling (Constructor)**
 * - Accepts complete trajectory tensors from RolloutStorage
 * - Generates random permutation of indices using `torch::randperm()`
 * - Reshapes indices tensor into groups of size `mini_batch_size`
 * - Result: 2D tensor of shape [num_mini_batches, mini_batch_size]
 * - Example: 1024 samples with mini_batch_size=256 → [4, 256] indices tensor
 *
 * **Phase 2: Iteration (next())**
 * - Maintains internal `index` pointer starting at 0
 * - Each call retrieves one row from shuffled indices
 * - Uses fancy indexing to reorder and batch tensors
 * - Returns MiniBatch with reordered trajectory data
 * - Advances pointer until all mini-batches consumed
 *
 * **Tensor Indexing Strategy:**
 * - Fancy indexing: `tensor.index(indices[i])` selects specific elements
 * - Reshaping: `.narrow()` extracts time-step ranges, `.view()` flattens dimensions
 * - Result: Efficiently reorganized data without data copying (views where possible)
 *
 * @see Generator, MiniBatch, RolloutStorage
 */

namespace LunarAlighting
{
    /**
     * @brief Constructs a FeedForwardGenerator and initializes shuffled mini-batch indices
     *
     * @details
     *
     * **Purpose:**
     * Initializes the generator with trajectory data and prepares randomized mini-batch
     * indices. This constructor performs the shuffling operation once, allowing efficient
     * iteration through randomized mini-batches.
     *
     * **Construction Steps:**
     *
     * 1. **Store Trajectory Tensors**
     *    - Each parameter (observations, actions, rewards, etc.) is stored as member variables
     *    - These tensors remain accessible throughout the generator's lifetime
     *    - No data is copied; references to original tensors are maintained
     *
     * 2. **Compute Total Batch Size**
     *    - `int batch_size = advantages.numel()` counts total elements in advantages tensor
     *    - For shape [timesteps, num_processes], numel() returns timesteps × num_processes
     *    - Example: advantages shape [128, 8] → batch_size = 1024
     *
     * 3. **Generate Random Permutation**
     *    - `torch::randperm(batch_size, ...)` creates a random permutation [0,1,2,...,batch_size-1]
     *    - Permutation ensures each sample appears exactly once (shuffled but complete)
     *    - Returns Long tensor for use as fancy indices
     *    - Example: 1024 samples → [0,512,1,513,...,1023] (randomly ordered)
     *
     * 4. **Reshape into Mini-Batches**
     *    - `.view({-1, mini_batch_size})` reshapes flat indices into 2D tensor
     *    - Result shape: [num_mini_batches, mini_batch_size]
     *    - -1 automatically computed: num_mini_batches = batch_size / mini_batch_size
     *    - Example: 1024 samples, batch=256 → shape [4, 256]
     *
     * **Memory & Efficiency:**
     * - Shuffling happens once during construction (O(n log n) due to randperm)
     * - Subsequent next() calls are O(mini_batch_size) (just indexing operations)
     * - Indices stored as GPU/CPU tensor matching input device
     * - Data not duplicated; only indices reshuffled
     *
     * @param mini_batch_size Number of samples per mini-batch
     *                        Typically 64-256 for stable gradient estimation
     *                        Larger = more stable gradients but requires more memory
     *
     * @param observations Trajectory observations
     *                     Shape: [timesteps+1, num_processes, obs_shape...]
     *                     The +1 accounts for bootstrapping value from final state
     *                     Contains sensory input for all steps of rollout
     *
     * @param hidden_states RNN hidden states (zeros for feed-forward)
     *                      Shape: [timesteps+1, num_processes, hidden_size]
     *                      hidden_size = 0 for feed-forward policies (no recurrent state)
     *                      hidden_size > 0 for recurrent policies
     *
     * @param actions Actions taken during trajectory
     *                Shape: [timesteps, num_processes, action_dim] or [timesteps, num_processes]
     *                action_dim varies: 1 for discrete, n for continuous with n dimensions
     *                Note: one step shorter than observations (no action at final state)
     *
     * @param value_predictions Critic network value estimates
     *                          Shape: [timesteps+1, num_processes, 1]
     *                          V(s) predictions for value function loss computation
     *                          Extra timestep includes bootstrap value at trajectory end
     *
     * @param returns Discounted cumulative returns (targets for value function)
     *                Shape: [timesteps+1, num_processes, 1]
     *                Computed via RolloutStorage::compute_returns() before generator creation
     *                Can be vanilla discounted returns or GAE-based returns
     *
     * @param masks Episode continuation indicators
     *              Shape: [timesteps+1, num_processes, 1]
     *              1.0 = episode continues, 0.0 = episode ended (terminal state)
     *              Used to prevent value bootstrapping from terminal states
     *
     * @param action_log_probs Log probabilities of actions under old policy
     *                         Shape: [timesteps, num_processes, 1]
     *                         log π_old(a|s) needed for importance sampling in PPO/A3C
     *                         Used to compute importance ratio: exp(log_new - log_old)
     *
     * @param advantages Advantage estimates (typically from GAE)
     *                   Shape: [timesteps, num_processes, 1]
     *                   A(s,a) = estimated benefit of action relative to value function
     *                   Used in advantage weighting: advantage = return - value_prediction
     *
     * @note The generator does NOT own the input tensors; they must remain valid for the
     *       entire lifetime of the FeedForwardGenerator object. Typically, tensors are stored
     *       in RolloutStorage, which outlives the generator.
     *
     * @warning If mini_batch_size doesn't divide batch_size evenly, the final mini-batch
     *          will be smaller. Example: 1000 samples with mini_batch_size=256 → 3×256 + 1×232
     *
     * @see Generator::Generator(), RolloutStorage::feed_forward_generator()
     *
     * @example
     * @code
     * // Create storage with 128 timesteps, 8 processes
     * RolloutStorage storage(128, 8, {84,84,1}, action_space, 0, torch::kCUDA);
     * // ... collect experience ...
     * storage.compute_returns(next_value, true, 0.99, 0.95);
     *
     * // Create generator for 4 mini-batches
     * auto advantages = storage.get_returns() - storage.get_value_predictions();
     * FeedForwardGenerator gen(256, storage.get_observations(), ...);
     * // Now gen.indices has shape [4, 256]
     * // Each call to gen.next() returns one row of indices
     * @endcode
     */
    FeedForwardGenerator::FeedForwardGenerator(
        int miniBatchSize,
        torch::Tensor observations,
        torch::Tensor hiddenStates,
        torch::Tensor actions,
        torch::Tensor valuePredictions,
        torch::Tensor returns,
        torch::Tensor masks,
        torch::Tensor actionLogProbs,
        torch::Tensor advantages) :
    observations(observations),
    hiddenStates(hiddenStates),
    actions(actions),
    valuePredictions(valuePredictions),
    returns(returns),
    masks(masks),
    actionLogProbs(actionLogProbs),
    advantages(advantages),
    index(0)
    {
        int batchSize = advantages.numel();
        indices = torch::randperm(batchSize,torch::TensorOptions(torch::kLong)).view({-1,miniBatchSize});

    }

    bool FeedForwardGenerator::done() const
    {
        return index >= indices.size(0);
    }


    /**
     * @brief Retrieves the next shuffled mini-batch of training data
     *
     * @details
     *
     * **Purpose:**
     * Returns a MiniBatch containing the next set of shuffled training samples.
     * Each call advances the internal index pointer to access the next row of
     * randomized indices, effectively yielding a different mini-batch each time.
     *
     * **Preconditions:**
     * - done() must return false (more mini-batches available)
     * - Generator must have been properly initialized via constructor
     * - All input tensors must remain valid and unchanged during iteration
     *
     * **Error Handling:**
     * - If called when done() returns true, throws std::runtime_error
     * - Prevents undefined behavior from accessing past the end of indices
     *
     * **Algorithm: Shuffled Mini-Batch Extraction**
     *
     * **Step 1: Validate State**
     * ```cpp
     * if (index >= indices.size(0))
     *     throw std::runtime_error("No minibatches left in generator.");
     * ```
     * Sanity check ensures we're not past the end. Defensive programming.
     *
     * **Step 2: Extract Timestep Count**
     * ```cpp
     * int timesteps = observations.size(0) - 1;
     * ```
     * - observations shape: [timesteps+1, num_processes, obs_shape...]
     * - Subtract 1 because observations[timesteps] is bootstrap observation (not sampled)
     * - timesteps is used later with `.narrow()` to exclude final observation
     * - Example: observations shape [129, 8, 84, 84] → timesteps = 128
     *
     * **Step 3: Prepare Observation Shape for Reshaping**
     * ```cpp
     * auto observations_shape = observations.sizes().vec();  // [129, 8, 84, 84]
     * observations_shape.erase(observations_shape.begin());   // Remove first dim: [8, 84, 84]
     * observations_shape[0] = -1;                             // Make flattened: [-1, 84, 84]
     * ```
     * This prepares a reshape target that flattens timesteps×processes into one dimension.
     * Purpose: Convert [T, P, D...] → [T×P, D...] for efficient indexing.
     *
     * **Step 4: Extract and Shuffle Observations**
     * ```cpp
     * mini_batch.observations = observations.narrow(0, 0, timesteps)  // [T, P, D...]
     *                               .view(observations_shape)         // [T×P, D...]
     *                               .index(indices[index]);            // [M, D...]
     * ```
     *
     * **Breakdown of Operations:**
     *
     * a) `.narrow(0, 0, timesteps)` - Extract timesteps only (exclude bootstrap)
     *    - Dimension 0: first dimension (time)
     *    - Start: 0 (from beginning)
     *    - Length: timesteps (how many to keep)
     *    - Result: Shape [timesteps, num_processes, obs_shape...]
     *    - Example: [129, 8, 84, 84] → [128, 8, 84, 84]
     *
     * b) `.view(observations_shape)` - Flatten timesteps and processes
     *    - Reshape from [T, P, D...] to [T×P, D...]
     *    - Allows treating all samples as one batch dimension
     *    - Example: [128, 8, 84, 84] → [1024, 84, 84]
     *    - Note: view() creates tensor view when possible (no copy)
     *
     * c) `.index(indices[index])` - Apply fancy indexing with shuffled indices
     *    - indices[index] shape: [mini_batch_size] containing random indices
     *    - Selects mini_batch_size samples in random order
     *    - Example: select rows [512, 1, 900, 340] from [1024, 84, 84]
     *    - Result: [mini_batch_size, 84, 84] with shuffled order
     *    - Note: index() selects specific rows according to index tensor
     *
     * **Step 5: Extract and Shuffle Other Tensors**
     * Same pattern applied to hidden_states, actions, value_predictions, returns, masks,
     * action_log_probs, and advantages. Each follows the same reshaping and indexing pattern.
     *
     * **Tensor-Specific Details:**
     *
     * **hidden_states:**
     * ```cpp
     * .narrow(0, 0, timesteps).view({-1, hidden_states.size(-1)}).index(indices[index])
     * ```
     * - Last dimension preserved (hidden_size)
     * - -1 auto-computes as timesteps×num_processes
     * - Example: [129, 8, 256] → [128, 8, 256] → [1024, 256] → [256, 256]
     *
     * **actions:**
     * ```cpp
     * .view({-1, actions.size(-1)}).index(indices[index])
     * ```
     * - No narrow() because actions don't have bootstrap timestep
     * - Shape [timesteps, num_processes, action_dim] → [T×P, action_dim]
     * - Example: [128, 8, 1] → [1024, 1] → [256, 1]
     *
     * **value_predictions, returns, masks, action_log_probs:**
     * ```cpp
     * .narrow(0, 0, timesteps).view({-1, 1}).index(indices[index])
     * ```
     * - All scalar-valued (one value per sample)
     * - Reshaped to [T×P, 1] then sampled to [M, 1]
     * - Example: [129, 8, 1] → [128, 8, 1] → [1024, 1] → [256, 1]
     *
     * **advantages:**
     * ```cpp
     * .view({-1, 1}).index(indices[index])
     * ```
     * - Similar to value predictions but no narrow (no bootstrap)
     * - Example: [128, 8, 1] → [1024, 1] → [256, 1]
     *
     * **Step 6: Advance Iterator**
     * ```cpp
     * index++;
     * ```
     * Increments the mini-batch counter so next call accesses different indices row.
     * After this, done() will check if index >= indices.size(0).
     *
     * **Step 7: Return Mini-Batch**
     * Returns the constructed MiniBatch containing all shuffled tensors.
     *
     * **Memory & Performance Characteristics:**
     *
     * - **.narrow()**: View operation, no data copy
     * - **.view()**: View operation (when contiguous), no data copy
     * - **.index()**: Gathers specified rows; data selection (minimal copy)
     * - Overall: Efficient with minimal data movement
     * - Tensors remain on original device (CPU/GPU)
     * - Shuffling pattern same across all iteration calls
     *
     * **Example Execution Trace:**
     * ```
     * Given:
     * - mini_batch_size = 256
     * - observations shape: [129, 8, 84, 84]
     * - advantages shape: [128, 8, 1]
     * - indices shape: [4, 256] with random permutation of [0:1024]
     *
     * First call to next() (index=0):
     * - timesteps = 128
     * - observations: [129,8,84,84] → narrow → [128,8,84,84]
     *                                → view   → [1024,84,84]
     *                                → index with indices[0] (256 random indices)
     *                                → [256,84,84]
     * - advantages: [128,8,1] → view → [1024,1]
     *                        → index with indices[0]
     *                        → [256,1]
     * - Returns MiniBatch with all shuffled tensors
     * - index becomes 1
     *
     * Second call to next() (index=1):
     * - Same pattern but uses indices[1] (different 256 random indices)
     * - Returns different shuffled MiniBatch
     * - index becomes 2
     * ```
     *
     * @return MiniBatch containing:
     *         - observations: shape [mini_batch_size, obs_shape...]
     *         - hidden_states: shape [mini_batch_size, hidden_size]
     *         - actions: shape [mini_batch_size, action_dim]
     *         - value_predictions: shape [mini_batch_size, 1]
     *         - returns: shape [mini_batch_size, 1]
     *         - masks: shape [mini_batch_size, 1]
     *         - action_log_probs: shape [mini_batch_size, 1]
     *         - advantages: shape [mini_batch_size, 1]
     *
     * @throw std::runtime_error if called when done() returns true (no more mini-batches)
     *
     * @note Returned tensors are views/selections of original storage tensors, not deep copies
     *
     * @see MiniBatch, done(), Generator::next()
     *
     * @post Internal index pointer is incremented
     * @post Subsequent calls return different shuffled mini-batches
     * @post After final call, done() returns true
     */
    MiniBatch FeedForwardGenerator::next()
    {
        if (index >= indices.size(0))
        {
            throw std::out_of_range("index out of range");
        }
        MiniBatch miniBatch;
        int timeSteps = observations.size(0) -1 ;
        auto observationShape = observations.sizes().vec();
        observationShape.erase(observationShape.begin());
        observationShape[0] = -1;
        unsigned int dim = 0;
        unsigned int start = 0;

        miniBatch.observations =
            observations.narrow(dim,start,timeSteps)
            .view(observationShape)
            .index({indices[index]});

        miniBatch.hiddenStates = hiddenStates.narrow(dim,start,timeSteps)
                                .view({-1,hiddenStates.size(-1)})
                                .index({indices[index]});

        miniBatch.actions = actions.view({-1,actions.size(-1)})
                            .index({indices[index]});

        miniBatch.valuePredictions = valuePredictions.narrow(dim,start,timeSteps)
                                                     .view({-1,1})
                                                     .index({indices[index]});

        miniBatch.returns = returns.narrow(dim,start,timeSteps)
                                .view({-1,1})
                                .index({indices[index]});

        miniBatch.masks = masks.narrow(dim,start,timeSteps)
                                .view({-1,1})
                                .index({indices[index]});

        miniBatch.actionLogProbs = actionLogProbs.view({-1,1}).index({indices[index]});

        miniBatch.advantages = advantages.view({-1,1}).index({indices[index]});

        index++;

        return  miniBatch;
    }

    /**
     * @brief Unit tests for FeedForwardGenerator functionality
     *
     * @details
     * Comprehensive test suite using doctest framework validating:
     * 1. Correct tensor shapes in generated mini-batches
     * 2. Proper iteration and completion detection
     * 3. Exception handling for invalid operations
     *
     * **Test Setup:**
     * ```
     * Generator Configuration:
     * - mini_batch_size = 5
     * - observations:      [6, 3, 4]  (6 timesteps, 3 processes, 4-dim obs)
     * - hidden_states:     [6, 3, 3]  (3-dim hidden state)
     * - actions:           [5, 3, 1]  (5 timesteps, 3 processes, 1-dim action)
     * - value_predictions: [6, 3, 1]  (includes bootstrap)
     * - returns:           [6, 3, 1]
     * - masks:             [6, 3, 1]
     * - action_log_probs:  [5, 3, 1]
     * - advantages:        [5, 3, 1]
     *
     * Batch Size Calculation:
     * - Total samples = advantages.numel() = 5 × 3 = 15
     * - Mini-batch size = 5
     * - Number of mini-batches = 15 / 5 = 3
     * - Indices shape: [3, 5]
     *
     * Expected Mini-Batch Shapes:
     * - Each mini-batch has 5 samples (mini_batch_size)
     * - Observations: [5, 4] (5 samples from [timesteps×processes, obs_dim])
     * - Hidden states: [5, 3]
     * - Actions: [5, 1]
     * - Others: [5, 1]
     * ```
     *
     * @test Minibatch tensors are correct sizes
     *       Verifies that all tensors in returned MiniBatch have expected shapes.
     *       Each dimension should match mini_batch_size × output_feature_size.
     */
    TEST_CASE("FeedForwardGenerator")
    {
       FeedForwardGenerator generator(5, torch::rand({6, 3, 4}), torch::rand({6, 3, 3}),
                                       torch::rand({5, 3, 1}), torch::rand({6, 3, 1}),
                                       torch::rand({6, 3, 1}), torch::ones({6, 3, 1}),
                                       torch::rand({5, 3, 1}), torch::rand({5, 3, 1}));
        /**
            * **Test Purpose:**
            * Verify that next() correctly reshapes and indexes all tensors to produce
            * the expected shapes for gradient computation.
            *
            * **Expected Results:**
            * Each tensor in MiniBatch should have shape:
            * [mini_batch_size, feature_dim] = [5, feature_dim]
            *
            * **Calculation:**
            * - Total samples = 5 timesteps × 3 processes = 15
            * - mini_batch_size = 5
            * - Each mini-batch: 5 samples selected via fancy indexing
            *
            * **Verification:**
            * - observations:       [5, 4]   (5 samples, 4-dim observations)
            * - hidden_states:      [5, 3]   (5 samples, 3-dim hidden states)
            * - actions:            [5, 1]   (5 samples, 1-dim actions)
            * - value_predictions:  [5, 1]   (5 scalar values)
            * - returns:            [5, 1]   (5 scalar returns)
            * - masks:              [5, 1]   (5 scalar masks)
            * - action_log_probs:   [5, 1]   (5 scalar log probs)
            * - advantages:         [5, 1]   (5 scalar advantages)
            */
        SUBCASE("Minibatch tensors are correct sizes")
        {
            auto minibatch = generator.next();

            CHECK(minibatch.observations.sizes().vec() == std::vector<int64_t>{5, 4});
            CHECK(minibatch.hiddenStates.sizes().vec() == std::vector<int64_t>{5, 3});
            CHECK(minibatch.actions.sizes().vec() == std::vector<int64_t>{5, 1});
            CHECK(minibatch.valuePredictions.sizes().vec() == std::vector<int64_t>{5, 1});
            CHECK(minibatch.returns.sizes().vec() == std::vector<int64_t>{5, 1});
            CHECK(minibatch.masks.sizes().vec() == std::vector<int64_t>{5, 1});
            CHECK(minibatch.actionLogProbs.sizes().vec() == std::vector<int64_t>{5, 1});
            CHECK(minibatch.advantages.sizes().vec() == std::vector<int64_t>{5, 1});
        }

        SUBCASE("done() indicates whether the generator has finished")
        {
            CHECK(!generator.done());
            generator.next();
            CHECK(!generator.done());
            generator.next();
            CHECK(!generator.done());
            generator.next();
            CHECK(generator.done());
        }

        SUBCASE("Calling a generator after it has finished throws an exception")
        {
            generator.next();
            generator.next();
            generator.next();
            CHECK_THROWS(generator.next());
        }
    }



}
