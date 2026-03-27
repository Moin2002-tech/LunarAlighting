"""
Pytorch-cpp-rl OpenAI gym server ZMQ client.
"""
import zmq
import msgpack
import logging


class ZmqClient:
    """
    Provides a ZeroMQ interface for communicating with client.
    """

    def __init__(self, port: int):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{port}")

    def receive(self) -> bytes:
        """
        Gets a message from the client.
        Blocks until a message is received.
        """
        message = self.socket.recv()
        try:
            response = msgpack.unpackb(message, raw=False)
        except msgpack.exceptions.ExtraData:
            response = message
        return response

    def send(self, message: object):
        """
        Sends a message to the client.
        """
        if isinstance(message, str):
            logging.info(f"Sending string message: {message}")
            self.socket.send_string(message)
        else:
            msg_bytes = message.to_msg()
            logging.info(f"Sending MessagePack message ({len(msg_bytes)} bytes): {msg_bytes[:100]}...")
            self.socket.send(msg_bytes)
