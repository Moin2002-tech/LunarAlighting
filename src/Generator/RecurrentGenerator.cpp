//
// Created by moinshaikh on 1/29/26.
//
#include<algorithm>
#include<vector>

#include<torch/torch.h>
#include<torch/nn.h>

#include"../../include/Generator/RecurrentGenerator.hpp"

#include <pstl/utils.h>

#include"../../include/Generator/Generator.hpp"

#include"../third_party/doctest.hpp"


namespace LunarAlighting
{
    /**
     * @brief Flattens a tensor by combining the first two dimensions (time and processes).
     *
     * @details This helper function reshapes a tensor of shape (timeStamps, processes, ...)
     * into a tensor of shape (timeStamps * processes, ...). This is commonly used to
     * flatten the batch dimensions for processing by neural networks or loss functions
     * that expect a single batch dimension.
     *
     * @param timeStamps The number of time steps in the data.
     * @param processes The number of parallel processes/environments.
     * @param tensors The input tensor to be flattened.
     * @return A new tensor with the first two dimensions flattened into one.
     */
    torch::Tensor flatten_helper(int timeStamps,int processes, torch::Tensor tensors)
    {
        auto tensorShape = tensors.sizes().vec();
        tensorShape.erase(tensorShape.begin());
        tensorShape[0] = timeStamps * processes;
        return tensors.view({tensorShape});
    }

    /**
     * @brief Implementation of the RecurrentGenerator constructor.
     *
     * @details Initializes the generator with collected experience data.
     * It calculates the number of environments per batch based on the total number of processes
     * and the desired number of mini-batches. It also generates a random permutation of
     * indices to shuffle the order of environments processed in each epoch.
     *
     * @param numProcesses Total number of parallel environments.
     * @param numMiniBatchSize Number of mini-batches to divide the data into.
     * @param observations Tensor of observations.
     * @param hiddenStates Tensor of hidden states.
     * @param actions Tensor of actions taken.
     * @param valuePredictions Tensor of value predictions.
     * @param returns Tensor of computed returns.
     * @param masks Tensor of masks (done flags).
     * @param actionLogProbs Tensor of action log probabilities.
     * @param advantages Tensor of computed advantages.
     */
    RecurrentGenerator::RecurrentGenerator(
        int numProcesses,
        int numMiniBatchSize,
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
    indices(torch::randperm(numProcesses,torch::TensorOptions(torch::kLong))),
    index(0),
    num_envs_per_batch(numProcesses / numMiniBatchSize)
    {

    }

    /**
     * @brief Checks if the generator has finished iterating through all mini-batches.
     *
     * @return true if the current index has reached or exceeded the number of indices,
     *         indicating all batches have been processed.
     */
    bool RecurrentGenerator::done() const
    {
        return index >= indices.size(/*dim*/0);
    }


    /**
     * @brief Generates the next mini-batch of experience.
     *
     * @details This function retrieves the next subset of environments based on the
     * shuffled indices. It slices the stored tensors (observations, hidden states, etc.)
     * to create a mini-batch.
     *
     * Key steps:
     * 1. Checks if there are more indices to process.
     * 2. Retrieves the current environment index from the shuffled list.
     * 3. Slices the data tensors for the selected environments.
     *    - Observations, value predictions, returns, masks, etc. are sliced for all time steps.
     *    - Hidden states are sliced only for the initial time step of the sequence.
     * 4. Flattens the time and process dimensions using `flatten_helper` to prepare
     *    the data for network input (batch_size = time_steps * num_processes_in_minibatch).
     * 5. Increments the internal index counter.
     *
     * @return A MiniBatch structure containing the flattened tensors for the current batch.
     * @throws std::runtime_error If called when no more indices are available.
     */
    MiniBatch RecurrentGenerator::next()
    {
        if (index >= indices.size(0))
        {
         throw std::runtime_error("Not enough indices");
        }

        MiniBatch mini_batch;

// Fill minibatch with tensors of shape (timestep, process, *whatever)
    // Except hidden states, that is just (process, *whatever)
    int64_t env_index = indices[index].item().toLong();
    mini_batch.observations = observations
                                  .narrow(0, 0, observations.size(0) - 1)
                                  .narrow(1, env_index, num_envs_per_batch);

    mini_batch.hiddenStates = hiddenStates[0]
                                   .narrow(0, env_index, num_envs_per_batch)
                                   .view({num_envs_per_batch, -1});

    mini_batch.actions = actions.narrow(1, env_index, num_envs_per_batch);

    mini_batch.valuePredictions = valuePredictions
                                       .narrow(0, 0, valuePredictions.size(0) - 1)
                                       .narrow(1, env_index, num_envs_per_batch);

    mini_batch.returns = returns.narrow(0, 0, returns.size(0) - 1)
                             .narrow(1, env_index, num_envs_per_batch);

    mini_batch.masks = masks.narrow(0, 0, masks.size(0) - 1)
                           .narrow(1, env_index, num_envs_per_batch);

    mini_batch.actionLogProbs = actionLogProbs.narrow(1, env_index,
                                                          num_envs_per_batch);
    mini_batch.advantages = advantages.narrow(1, env_index, num_envs_per_batch);

        // Flatten tensors to (timestep * process, *whatever)
        int num_timesteps = mini_batch.observations.size(0);
        int num_processes = num_envs_per_batch;
        mini_batch.observations = flatten_helper(num_timesteps, num_processes,
                                                mini_batch.observations);
        mini_batch.actions = flatten_helper(num_timesteps, num_processes,
                                            mini_batch.actions);
        mini_batch.valuePredictions = flatten_helper(num_timesteps, num_processes,
                                                    mini_batch.valuePredictions);
        mini_batch.returns = flatten_helper(num_timesteps, num_processes,
                                            mini_batch.returns);
        mini_batch.masks = flatten_helper(num_timesteps, num_processes,
                                        mini_batch.masks);
        mini_batch.actionLogProbs = flatten_helper(num_timesteps, num_processes,
                                                    mini_batch.actionLogProbs);
        mini_batch.advantages = flatten_helper(num_timesteps, num_processes,
                                            mini_batch.advantages);


        index++;
        return mini_batch;
    }

    TEST_CASE("RecurrentGenerator")
    {
        RecurrentGenerator generator(3, 3, torch::rand({6, 3, 4}),
                                     torch::rand({6, 3, 3}), torch::rand({5, 3, 1}), torch::rand({6, 3, 1}), torch::rand({6, 3, 1}), torch::ones({6, 3, 1}), torch::rand({5, 3, 1}), torch::rand({5, 3, 1}));

        SUBCASE("Minibatch tensors are correct sizes")
        {
            auto minibatch = generator.next();

            CHECK(minibatch.observations.sizes().vec() == std::vector<int64_t>{5, 4});
            CHECK(minibatch.hiddenStates.sizes().vec() == std::vector<int64_t>{1, 3});
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
