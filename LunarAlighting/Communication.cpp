//
// Created by moinshaikh on 2/9/26.
//

#include<memory>
#include<string>

#include"Communication.hpp"

#include<spdlog/spdlog.h>
#include"Request.hpp"
#include"third_party/zmq.hpp"

namespace GymClient
{
    Communicator::Communicator(const std::string &url)
    {
        context = std::make_unique<zmq::context_t>(1);
        socket = std::make_unique<zmq::socket_t>(*context,ZMQ_PAIR);

        // Set receive timeout to 5 seconds (5000 milliseconds)
        int timeout = 5000;
        socket->setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
        
        socket->connect(url.c_str());
        spdlog::info("Connected to gym environment at: {}", url);
    }

    Communicator::~Communicator() {

    }

    std::string Communicator::getRawResponse()
    {
        zmq::message_t zmqMessage;
        socket->recv(&zmqMessage);

        std::string response =  std::string (static_cast<char * >(zmqMessage.data()), zmqMessage.size());


        return response;
    }
}
