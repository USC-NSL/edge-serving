#!/usr/bin/env python2.7
from __future__ import print_function

import grpc
import threading
import time
from concurrent import futures

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/')
from modules.Tacotron import Tacotron
from modules.audio_resample import Resample
from modules.Deepspeech2 import Deepspeech2
from modules.Jasper import Jasper
from modules.Wave2Letter import Wave2Letter
from modules.text_encoder import TextEncoder
from modules.Transformer import Transformer
from modules.TransformerBig import TransformerBig
from modules.Convs2s import Convs2s
from modules.text_decoder import TextDecoder

from tensorflow.python.framework import tensor_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import olympian_master_pb2_grpc
from tensorflow_serving.apis import olympian_worker_pb2_grpc
from tensorflow_serving.apis import olympian_client_pb2_grpc

import logging
logging.basicConfig()

from chain_modules.chain_mobilenet_v3 import MobilenetPreprocess
from chain_modules.chain_mobilenet_v3 import MobilenetInference

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

tf.app.flags.DEFINE_string('worker', 'localhost:50101', 'Olympian worker host:port')
FLAGS = tf.app.flags.FLAGS

# Worker Class
class OlympianWorker(olympian_worker_pb2_grpc.OlympianWorkerServicer):

  def __init__(self):
    self.cstubs = dict()

    # self.worker_list = ["localhost:50101", "localhost:50102"]
    self.worker_list = ["localhost:50101"]
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

  def parseRouteTable(self, route_table, route_index):
    tmp = route_table.split("-")
    current_model = tmp[route_index].split(":")[0]
    tt = tmp[route_index + 1].split(":")
    next_model = tt[0]
    next_stub = "%s:%s" % (tt[1], tt[2])
    return current_model, next_model, next_stub

  def Predict(self, request, context):
    if (request.model_spec.signature_name == "chain_specification"): # gRPC from client
      chain_name = request.model_spec.name
      frame_info = tensor_util.MakeNdarray(request.inputs["frame_info"])
      client_address = str(frame_info).split('-')[1]

      route_table = str(tensor_util.MakeNdarray(request.inputs["route_table"]))
      route_index = int(tensor_util.MakeNdarray(request.inputs["route_index"]))

      current_model, _, next_stub = self.parseRouteTable(route_table, route_index)

      print("========== Predict() ==========")
      print("[%s][Worker] Received request using chain %s" % (str(time.time()), chain_name))
      print("[%s][Worker] current_model = %s" % (time.time(), current_model))
      print("                        next_stub = %s" % (next_stub))
      print("                        frame_info = %s" % (frame_info))

      if (current_model == "exported_mobilenet_v1_1.0_224_preprocess"):
        module_instance = MobilenetPreprocess()

      elif (current_model == "exported_mobilenet_v1_1.0_224_inference"):
        module_instance = MobilenetInference()

      elif (current_model == "tacotron"):
        module_instance = Tacotron()

      elif (current_model == "nlpCPU"):
        module_instance = Resample()

      elif (current_model == "speech2text"):
        module_instance = Deepspeech2()

      elif (current_model == "jasper"):
        module_instance = Jasper()

      elif (current_model == "wave2letter"):
        module_instance = Wave2Letter()

      elif (current_model == "encoder"):
        module_instance = TextEncoder()

      elif (current_model == "transformer"):
        module_instance = Transformer()

      elif (current_model == "transformer_big"):
        module_instance = TransformerBig()

      elif (current_model == "conv_s2s"):
        module_instance = Convs2s()

      elif (current_model == "decoder"):
        module_instance = TextDecoder()

      else:
        print("[Worker] Error...")

      module_instance.Setup()
      module_instance.PreProcess(request, self.istub)
      module_instance.Apply()
      next_request = module_instance.PostProcess()

      print("[%s][Worker] Received result from local TF-Serving, ready for next stub %s\n" % (str(time.time()), next_stub))
      next_request.model_spec.name = chain_name
      next_request.model_spec.signature_name = "chain_specification"

      next_request.inputs['frame_info'].CopyFrom(
        tf.make_tensor_proto(str(frame_info)))
      next_request.inputs['route_table'].CopyFrom(
        tf.make_tensor_proto(str(route_table)))
      next_request.inputs['route_index'].CopyFrom(
        tf.make_tensor_proto(route_index + 1, dtype=tf.int32))

      if (next_stub == client_address):
        fchannel = grpc.insecure_channel(next_stub)
        fstub = olympian_client_pb2_grpc.OlympianClientStub(fchannel)
        next_result = fstub.Predict(next_request, 30.0)
        fchannel.close()
      else:
        next_result = self.cstubs[next_stub].Predict(next_request, 30.0)

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