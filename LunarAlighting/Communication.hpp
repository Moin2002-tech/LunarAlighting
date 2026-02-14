#pragma once
/**
 * @file Communication.hpp
 * @brief ZeroMQ-based communication client for gym environment interaction
 * @author moinshaikh
 * @date 2/9/26
 * 
 * This file implements a communication client using ZeroMQ for interacting with
 * gym environment servers. It provides templated methods for sending requests
 * and receiving responses using MessagePack serialization.
 * 
 * Key features:
 * - ZeroMQ socket-based communication
 * - MessagePack serialization/deserialization
 * - Template-based request/response handling
 * - Error handling and logging
 * - RAII resource management
 */

#ifndef LUNARALIGHTINGRL_COMMUNICATION_HPP
#define LUNARALIGHTINGRL_COMMUNICATION_HPP

#include<iostream>
#include<memory>
#include<sstream>
#include<string>

#include<msgpack.hpp>
#include<spdlog/spdlog.h>
#include<fmt/ostream.h>

#include"Request.hpp"
#include"third_party/zmq.hpp"

/**
 * @namespace GymClient
 * @brief Client-side communication utilities for gym environment interaction
 * 
 * This namespace contains communication components for interacting with gym environment
 * servers. It provides high-level abstractions for sending requests and receiving
 * responses using ZeroMQ messaging and MessagePack serialization.
 */
namespace GymClient
{
    /**
     * @class Communicator
     * @brief ZeroMQ-based communication client for gym environment servers
     * 
     * This class provides a high-level interface for communicating with gym environment
     * servers using ZeroMQ sockets and MessagePack serialization. It handles both
     * raw string responses and typed deserialization using template methods.
     * 
     * The class follows RAII principles for resource management and includes error
     * handling for communication failures and deserialization errors.
     * 
     * Usage pattern:
     * 1. Create Communicator with server URL
     * 2. Send requests using sendRequest<T>()
     * 3. Receive responses using getResponse<T>() or getRawResponse()
     * 4. Automatic cleanup on destruction
     */
    class Communicator
    {
    private:
        std::unique_ptr<zmq::context_t> context; ///< ZeroMQ context for socket management
        std::unique_ptr<zmq::socket_t> socket;   ///< ZeroMQ socket for communication
        
    public:
        /**
         * @brief Construct a new Communicator object
         * @param url The ZeroMQ server URL (e.g., "tcp://localhost:5555")
         * 
         * Initializes ZeroMQ context and socket, then connects to the specified
         * server URL. The socket is configured for request-reply pattern.
         * 
         * @throws zmq::error_t if socket creation or connection fails
         */
        Communicator(const std::string &url);
        
        /**
         * @brief Destroy the Communicator object
         * 
         * Properly closes the ZeroMQ socket and destroys the context.
         * Ensures clean shutdown of communication resources.
         */
        ~Communicator();

        /**
         * @brief Receive raw string response from server
         * @return Raw response string from the server
         * 
         * This method receives a message from the ZeroMQ socket and returns it
         * as a raw string. Useful for debugging or when response format
         * is unknown or doesn't require deserialization.
         * 
         * @note This method blocks until a message is received
         * @throws zmq::error_t if receiving fails
         */
        std::string getRawResponse();

        /**
         * @brief Receive and deserialize typed response from server
         * @tparam T The response type to deserialize into
         * @return Unique pointer to deserialized response object
         * 
         * This template method receives a MessagePack-encoded message from the server
         * and deserializes it into the specified type. It handles deserialization
         * errors gracefully by logging the error and returning a default-constructed object.
         * 
         * Process:
         * 1. Receive raw message from ZeroMQ socket
         * 2. Unpack MessagePack data
         * 3. Convert to specified type T
         * 4. Handle deserialization errors with logging
         * 
         * @tparam T Response type must be MessagePack-serializable
         * @note This method blocks until a message is received
         * @throws zmq::error_t if receiving fails
         * @warning On deserialization error, logs error and returns default-constructed object
         */
        template <typename T>
        std::unique_ptr<T> getResponse()
        {
            // Receive packed message from ZeroMQ socket
            zmq::message_t packedMessage;
            try {
                socket->recv(&packedMessage);
            } catch (const zmq::error_t& e) {
                if (e.num() == EAGAIN) {
                    // Timeout occurred
                    spdlog::error("Timeout waiting for response from gym server");
                    return nullptr;
                }
                // Other ZMQ errors
                spdlog::error("ZMQ error receiving response: {}", e.what());
                return nullptr;
            }
            
            // Unpack MessagePack data
            msgpack::object_handle objectHandle = msgpack::unpack(static_cast<char * >(packedMessage.data()),packedMessage.size());

            // Extract MessagePack object
            msgpack::object object = objectHandle.get();
            
            // Create response object and deserialize
            std::unique_ptr<T> response = std::make_unique<T>();
            try
            {
               // Convert MessagePack object to specified type
               object.convert(response);
            }
            catch (...)
            {
                // Log deserialization error with object details
                spdlog::error("Communication error {}", object.as<std::string>());
                return nullptr;
            }
            return response;
        }

        /**
         * @brief Serialize and send request to server
         * @tparam T The request parameter type
         * @param request The Request object containing method and parameters
         * 
         * This template method serializes a Request object using MessagePack and
         * sends it to the connected gym environment server. The Request object
         * contains the method name and parameters for the gym operation.
         * 
         * Process:
         * 1. Serialize Request object to MessagePack buffer
         * 2. Create ZeroMQ message from buffer data
         * 3. Send message to server
         * 
         * @tparam T Request parameter type must be MessagePack-serializable
         * @note This method blocks until the message is sent
         * @throws zmq::error_t if sending fails
         * @throws msgpack::type_error if serialization fails
         */
        template<class T>
        void sendRequest(const Request<T> &request)
        {
            // Serialize request to MessagePack buffer
            msgpack::sbuffer buffer;
            msgpack::pack(buffer, request);

            // Create ZeroMQ message from serialized data
            zmq::message_t message(buffer.size());
            std::memcpy(message.data(), buffer.data(), buffer.size());
            
            // Send message to server
            socket->send(message);
        }
    };
}

#endif //LUNARALIGHTINGRL_COMMUNICATION_HPP