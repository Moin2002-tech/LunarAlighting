 #pragma once
/**
 * @file Request.hpp
 * @brief Request and response structures for GymClient communication
 * @author moinshaikh
 * @date 2/9/26
 * 
 * This file defines the communication protocol structures used for interacting
 * with gym environments through the GymClient namespace. It includes parameter
 * structures for different gym operations and corresponding response structures
 * for both CNN and MLP observation types.
 */

#ifndef LUNARALIGHTINGRL_REQUEST_HPP
#define LUNARALIGHTINGRL_REQUEST_HPP

#include<msgpack.hpp>

#include<string>


/**
 * @namespace GymClient
 * @brief Client-side structures for gym environment communication
 * 
 * This namespace contains all data structures required for communication
 * between the client and gym environment server. It includes request templates,
 * parameter structures for different operations, and response structures for
 * various observation types (CNN and MLP).
 */
namespace GymClient
{
    /**
     * @template Request
     * @brief Generic request template for gym operations
     * @tparam T The parameter type for the specific gym operation
     * 
     * This template structure provides a standardized format for sending requests
     * to the gym environment. Each request contains a method name and a shared
     * pointer to the parameter structure specific to that operation.
     * 
     * The use of shared_ptr ensures efficient memory management and allows
     * for safe sharing of parameter data between different components.
     */
    template<class T>
    struct Request
    {
        std::string method; ///< The name of the gym operation to perform
        std::shared_ptr<T> param; ///< Shared pointer to operation-specific parameters
        
        /**
         * @brief Construct a new Request object
         * @param method The gym operation method name (e.g., "info", "make", "reset", "step")
         * @param param Shared pointer to the parameter structure for the operation
         * 
         * This constructor initializes a request with the specified method name
         * and parameter structure. The parameters are stored as a shared pointer
         * to enable efficient memory management and data sharing.
         */
        Request(const std::string &method, std::shared_ptr<T> param) : method(method), param(param)
        {

        }
        
        MSGPACK_DEFINE_MAP(method, param); ///< MessagePack serialization definition
    };

    /**
     * @struct infoParam
     * @brief Parameters for gym environment info request
     * 
     * This structure contains parameters needed to request information
     * about a specific gym environment instance. The integer 'x' likely
     * represents the environment ID or instance identifier.
     */
    struct infoParam
    {
        int x; ///< Environment instance identifier
        MSGPACK_DEFINE_MAP(x); ///< MessagePack serialization definition
    };

    /**
     * @struct makeParam
     * @brief Parameters for creating new gym environment instances
     * 
     * This structure contains parameters needed to create one or more
     * instances of a specific gym environment. It specifies the environment
     * name and the number of parallel environments to create.
     */
    struct makeParam
    {
        std::string envName; ///< Name of the gym environment to create
        int numEnv; ///< Number of parallel environment instances to create
        MSGPACK_DEFINE_MAP(envName,numEnv); ///< MessagePack serialization definition
    };

    /**
     * @struct resetParam
     * @brief Parameters for resetting gym environment instances
     * 
     * This structure contains parameters needed to reset a specific
     * gym environment instance to its initial state. The integer 'x'
     * represents the environment instance identifier.
     */
    struct resetParam
    {
        int x; ///< Environment instance identifier to reset
        MSGPACK_DEFINE_MAP(x); ///< MessagePack serialization definition
    };

    /**
     * @struct stepParam
     * @brief Parameters for stepping gym environment instances
     * 
     * This structure contains parameters needed to execute one or more
     * actions in the gym environment. It supports batched actions for
     * multiple parallel environments and optional rendering.
     */
    struct stepParam
    {
        std::vector<std::vector<float>> action; ///< Batched actions for all parallel environments
        bool render; ///< Whether to render the environment after taking the action
        MSGPACK_DEFINE_MAP(action,render); ///< MessagePack serialization definition
    };

    /**
     * @struct InfoResponse
     * @brief Response structure containing gym environment information
     * 
     * This structure contains detailed information about the gym environment's
     * action and observation spaces, including their types and shapes.
     */
    struct InfoResponse
    {
        std::string actionSpaceType; ///< Type of the action space (e.g., "Discrete", "Box", "MultiDiscrete")
        std::vector<int64_t> actionSpaceShape; ///< Shape/dimensions of the action space
        std::string observationSpaceType; ///< Type of the observation space (e.g., "Box", "Discrete")
        std::vector<int64_t> observationSpaceShape; ///< Shape/dimensions of the observation space
        MSGPACK_DEFINE_MAP(actionSpaceType,actionSpaceShape,observationSpaceType,observationSpaceShape); ///< MessagePack serialization definition
    };

    /**
     * @struct MakeResponse
     * @brief Response structure for environment creation requests
     * 
     * This structure contains the result of a gym environment creation request,
     * typically indicating success or failure status.
     */
    struct MakeResponse
    {
        std::string result; ///< Result message indicating success or failure of environment creation
        MSGPACK_DEFINE_MAP(result); ///< MessagePack serialization definition
    };

    /**
     * @struct CnnResetResponse
     * @brief Response structure for CNN-based environment reset
     * 
     * This structure contains the initial observations after resetting a CNN-based
     * gym environment. The observations are 4D tensors suitable for convolutional
     * neural networks.
     */
    struct CnnResetResponse
    {
        std::vector<std::vector<std::vector<std::vector<float>>>> observation; ///< 4D observation tensor [batch][channels][height][width]
        MSGPACK_DEFINE_MAP(observation); ///< MessagePack serialization definition
    };

    /**
     * @struct MlpResetResponse
     * @brief Response structure for MLP-based environment reset
     * 
     * This structure contains the initial observations after resetting an MLP-based
     * gym environment. The observations are 2D matrices suitable for multi-layer
     * perceptrons.
     */
    struct MlpResetResponse
    {
        std::vector<std::vector<float>> observation; ///< 2D observation matrix [batch][features]
        MSGPACK_DEFINE_MAP(observation); ///< MessagePack serialization definition
    };

    /**
     * @struct StepResponse
     * @brief Base response structure for environment step operations
     * 
     * This structure contains the common response data for all environment step
     * operations, including rewards, completion flags, and real rewards.
     */
    struct StepResponse
    {
        std::vector<std::vector<float>> reward; ///< Rewards received for each environment step
        std::vector<std::vector<bool>> done; ///< Completion flags indicating if episodes are finished
        std::vector<std::vector<float>> real_reward; ///< Actual rewards without potential shaping
    };

    /**
     * @struct CnnStepResponse
     * @brief Response structure for CNN-based environment step operations
     * 
     * This structure extends StepResponse to include CNN-compatible observations
     * after executing actions in the environment. Suitable for environments with
     * image-based observations.
     */
    struct CnnStepResponse : StepResponse
    {
        std::vector<std::vector<std::vector<std::vector<float>>>> observation; ///< 4D observation tensor [batch][channels][height][width]
        MSGPACK_DEFINE_MAP(observation, reward, done, real_reward); ///< MessagePack serialization definition
    };

    /**
     * @struct MlpStepResponse
     * @brief Response structure for MLP-based environment step operations
     * 
     * This structure extends StepResponse to include MLP-compatible observations
     * after executing actions in the environment. Suitable for environments with
     * vector-based observations.
     */
    struct MlpStepResponse : StepResponse
    {
        std::vector<std::vector<float>> observation; ///< 2D observation matrix [batch][features]
        MSGPACK_DEFINE_MAP(observation, reward, done, real_reward); ///< MessagePack serialization definition
    };
}

#endif //LUNARALIGHTINGRL_REQUEST_HPP