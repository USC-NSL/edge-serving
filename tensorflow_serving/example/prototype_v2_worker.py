#!/usr/bin/env python2.7
from __future__ import print_function

import grpc
import threading
import time
from concurrent import futures

import tensorflow as tf

from tensorflow.python.framework import tensor_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import olympian_master_pb2_grpc
from tensorflow_serving.apis import olympian_worker_pb2_grpc
from tensorflow_serving.apis import olympian_client_pb2_grpc

import logging
logging.basicConfig()

from chain_modules.chain_mobilenet_v2 import MobilenetPreprocess
from chain_modules.chain_mobilenet_v2 import MobilenetInference

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

tf.app.flags.DEFINE_string('worker', 'localhost:50100', 'Olympian worker host:port')
FLAGS = tf.app.flags.FLAGS

# Worker Class
class OlympianWorker(olympian_worker_pb2_grpc.OlympianWorkerServicer):

  def __init__(self):
    self.cstubs = dict()

    self.worker_list = ["localhost:50101", "localhost:50102"]
    for w in self.worker_list:
      channel = grpc.insecure_channel(w)
      stub = olympian_worker_pb2_grpc.OlympianWorkerStub(channel)
      self.cstubs[w] = stub

    self.master_list = ["localhost:50051"]
    for m in self.master_list:
      channel = grpc.insecure_channel(m)
      stub = olympian_master_pb2_grpc.OlympianMasterStub(channel)
      self.cstubs[m] = stub

    ichannel = grpc.insecure_channel("localhost:8500")
    self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

  def parseRouteTable(self, route_table, worker_address):
    tmp = route_table.split("-")
    for i in range(len(tmp)):
      t = tmp[i].split(":")
      tstub = "%s:%s" % (t[1], t[2])
      if (tstub == worker_address):
        current_model = t[0]
        tt = tmp[i + 1].split(":")
        next_stub = "%s:%s" % (tt[1], tt[2])
        return current_model, next_stub
    return "Error", "Error"

  def Predict(self, request, context):
    if (request.model_spec.signature_name == "chain_specification"): # gRPC from client
      route_table = str(tensor_util.MakeNdarray(request.inputs["route_table"]))
      current_model, next_stub = self.parseRouteTable(route_table, FLAGS.worker)

      if (current_model == "exported_mobilenet_v1_1.0_224_preprocess"):
        mobilenetpreprocess = MobilenetPreprocess()
        mobilenetpreprocess.Setup()
        mobilenetpreprocess.PreProcess(request, route_table, current_model, next_stub, self.istub, self.cstubs[next_stub])
        mobilenetpreprocess.Apply()
        mobilenetpreprocess.PostProcess()

      elif (current_model == "exported_mobilenet_v1_1.0_224_inference"):
        fchannel = grpc.insecure_channel(next_stub)
        fstub = olympian_client_pb2_grpc.OlympianClientStub(fchannel)

        mobilenetinference = MobilenetInference()
        mobilenetinference.Setup()
        mobilenetinference.PreProcess(request, route_table, current_model, next_stub, self.istub, fstub)
        mobilenetinference.Apply()
        mobilenetinference.PostProcess()

        fchannel.close()

      else:
        print("[Worker] Error...")

      dumbresult = predict_pb2.PredictResponse()
      dumbresult.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))
      return dumbresult

    else: # Not sure yet...
      print("[Worker] Not sure yet...")
      dumbresult = predict_pb2.PredictResponse()
      dumbresult.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))
      return dumbresult

def main(_):
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
  olympian_worker_pb2_grpc.add_OlympianWorkerServicer_to_server(OlympianWorker(), server)
  server.add_insecure_port(FLAGS.worker)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  tf.app.run()