# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

# from grpc.beta import implementations
import tensorflow as tf

from tensorflow.python.framework import tensor_util

# from tensorflow_serving.apis import tomtest_pb2
# from tensorflow_serving.apis import tomtest_grpc_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import olympian_master_pb2_grpc
from tensorflow_serving.apis import olympian_worker_pb2_grpc

import time
import numpy as np

import logging
logging.basicConfig()

from concurrent import futures
import grpc
import threading

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

tf.app.flags.DEFINE_string('worker', 'localhost:50100',
                           'Olympian worker host:port')
FLAGS = tf.app.flags.FLAGS

from chain_modules.chain_mobilenet_v1 import MobilenetPreprocess
from chain_modules.chain_mobilenet_v1 import MobilenetInference

# Worker Class
class OlympianWorker(olympian_worker_pb2_grpc.OlympianWorkerServicer):

  def sendHeartBeat(self):
    # send heartbeat message to Master every 15 sec.
    while (True):
      hb_message = "Hello from %s" % str(FLAGS.worker)

      hb_request = predict_pb2.PredictRequest()
      hb_request.model_spec.name = "unknown"
      hb_request.model_spec.signature_name = 'unknown'

      hb_request.inputs['HEARTBEAT'].CopyFrom(
        tf.make_tensor_proto(hb_message))
      try:
        hb_result = self.cstubs[self.master_list[0]].Predict(hb_request, 10.0)
      except Exception as e:
        print("Failed with: %s" % str(e)) 
        break

      time.sleep(15)

  def __init__(self):
    self.cstubs = dict()

    # add worker stub
    self.worker_list = ["localhost:50101", "localhost:50102"]
    # worker_list = ["192.168.1.125:50101", "192.168.1.102:50102"]
    for w in self.worker_list:
      channel = grpc.insecure_channel(w)
      stub = olympian_worker_pb2_grpc.OlympianWorkerStub(channel)
      self.cstubs[w] = stub
    # add master stub
    self.master_list = ["localhost:50051"]
    # master_list = ["192.168.1.102:50051"]
    for m in self.master_list:
      channel = grpc.insecure_channel(m)
      stub = olympian_master_pb2_grpc.OlympianMasterStub(channel)
      self.cstubs[m] = stub

    # add istub for internal TF-Serving
    ichannel = grpc.insecure_channel("localhost:9000")
    self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

    # send heartbeat periodically
    t = threading.Thread(target = self.sendHeartBeat)
    t.start()

  def getStubInfo(self, route_table, current_stub):
    tmp = route_table.split("-")
    for i in range(len(tmp)):
      t = tmp[i]
      tt = t.split(":")
      tstub = "%s:%s" % (tt[1], tt[2])
      if (tstub == current_stub):
        current_model = tt[0]
        ttt = tmp[i + 1]
        tttt = ttt.split(":")
        next_stub = "%s:%s" % (tttt[1], tttt[2])
        return current_model, next_stub
    return "Error", "Error"

  def printRouteTable(self, route_table, machine_name):
    tmp = route_table.split("-")
    for i in range(len(tmp)):
      print("[%s][%s] route info: hop-%s %s" % (str(time.time()), machine_name, str(i).zfill(2), tmp[i]))

  def Predict(self, request, context):
    if (request.model_spec.signature_name == "chain_specification"): # gRPC from client

      chain_name = request.model_spec.name
      route_table = tensor_util.MakeNdarray(request.inputs["route_table"])
      current_model, next_stub = self.getStubInfo(str(route_table), FLAGS.worker)

      if (current_model == "exported_mobilenet_v1_1.0_224_preprocess"):
        mobilenetpreprocess = MobilenetPreprocess()
        mobilenetpreprocess.Setup(chain_name, route_table, current_model, next_stub, request, self.istub, self.cstubs[next_stub])
        mobilenetpreprocess.PreProcess()
        mobilenetpreprocess.Apply()
        mobilenetpreprocess.PostProcess()

      elif (current_model == "exported_mobilenet_v1_1.0_224_inference"):
        mobilenetinference = MobilenetInference()
        mobilenetinference.Setup(chain_name, route_table, current_model, next_stub, request, self.istub, self.cstubs[next_stub])
        mobilenetinference.PreProcess()
        mobilenetinference.Apply()
        mobilenetinference.PostProcess()

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