#!/usr/bin/env python2.7
from __future__ import print_function

import grpc
import time
from concurrent import futures

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.framework import tensor_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import olympian_master_pb2_grpc
from tensorflow_serving.apis import olympian_worker_pb2_grpc

import logging
logging.basicConfig()

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

# Master Class
class OlympianMaster(olympian_master_pb2_grpc.OlympianMasterServicer):

  def __init__(self):
    self.cstubs = dict()
    worker_list = ["localhost:50101", "localhost:50102"]
    for w in worker_list:
      channel = grpc.insecure_channel(w)
      stub = olympian_worker_pb2_grpc.OlympianWorkerStub(channel)
      self.cstubs[w] = stub
    master_list = ["localhost:50051"]
    for m in master_list:
      channel = grpc.insecure_channel(m)
      stub = olympian_master_pb2_grpc.OlympianMasterStub(channel)
      self.cstubs[m] = stub

  def getRouteTable(self, chain_name, client_address):
    if (chain_name == "chain_mobilenet"):
      base_route_table = "exported_mobilenet_v1_1.0_224_preprocess:localhost:50101-exported_mobilenet_v1_1.0_224_inference:localhost:50102"
      route_table = "%s-FINAL:%s" % (base_route_table, client_address)
      return route_table
    elif (chain_name == "chain_nlp"):
      base_route_table = "saved_models:localhost:50101-nlpCPU:localhost:50102-speech2text:localhost:50102"
      # base_route_table = "saved_models:localhost:50101-nlpCPU:localhost:50102-jasper:localhost:50102"
      # base_route_table = "saved_models:localhost:50101-nlpCPU:localhost:50102-wave2letter:localhost:50102"
      route_table = "%s-FINAL:%s" % (base_route_table, client_address)
      return route_table
    else:
      return "Error, something is wrong..."

  def Predict(self, request, context):
    print("========== Predict() ==========")
    if ("sess_setup" in request.inputs): # gRPC of sess id request from client
      chain_name = str(request.model_spec.name)
      client_address = str(tensor_util.MakeNdarray(request.inputs["sess_setup"]))

      print("[%s][Master] Received sess id request for %s w/ client_address = %s, peer = %s\n" % (str(time.time()), chain_name, client_address, context.peer()))
      route_table = self.getRouteTable(chain_name, client_address)

      sess_id = "%s-%s" % (chain_name, client_address)
      sess_setup_response = predict_pb2.PredictResponse()
      sess_setup_response.outputs["sess_id"].CopyFrom(
        tf.make_tensor_proto(sess_id))
      sess_setup_response.outputs["route_table"].CopyFrom(
        tf.make_tensor_proto(route_table))
      return sess_setup_response

    else:
      print("[%s][Master] Something is wrong..." % (str(time.time())))

    # default dumb OK message...
    dumbresult = predict_pb2.PredictResponse()
    dumbresult.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))
    return dumbresult


def main(_):
  
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
  olympian_master_pb2_grpc.add_OlympianMasterServicer_to_server(OlympianMaster(), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  tf.app.run()