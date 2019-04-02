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

sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/')
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/modules_actdet/SSD-Tensorflow')
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/modules_actdet/deep_sort')
from modules_actdet.data_reader import DataReader
from modules_actdet.object_detector_ssd import SSD
from modules_actdet.object_detector_yolo import YOLO
from modules_actdet.tracker_deepsort import DeepSort
from modules_actdet.action_detector_acam import ACAM

from tensorflow.python.framework import tensor_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import olympian_master_pb2_grpc
from tensorflow_serving.apis import olympian_worker_pb2_grpc
from tensorflow_serving.apis import olympian_client_pb2_grpc

from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2
from google.protobuf import text_format

import logging
logging.basicConfig()

from chain_modules.chain_mobilenet_v3 import MobilenetPreprocess
from chain_modules.chain_mobilenet_v3 import MobilenetInference

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64
MAX_WORKERS = 600

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
    self.istub_reload = model_service_pb2_grpc.ModelServiceStub(ichannel)

    self.loaded_model_set = set()

    self.model_path_dict = dict()
    self.model_path_dict['tacotron'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/tacotron'
    self.model_path_dict['deepspeech2'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/deepspeech2'
    self.model_path_dict['transformer'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/transformer'
    self.model_path_dict['jasper'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/jasper'
    self.model_path_dict['wave2letter'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/wave2letter'
    self.model_path_dict['conv_s2s'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/conv_s2s'
    self.model_path_dict['transformer_big'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/transformer_big'
    self.model_path_dict['exported_mobilenet_v1_1.0_224_preprocess'] = '/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/exported_mobilenet_v1_1.0_224_preprocess'
    self.model_path_dict['exported_mobilenet_v1_1.0_224_inference'] = '/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/exported_mobilenet_v1_1.0_224_inference'
    self.model_path_dict['actdet_ssd'] = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/actdet_ssd'
    self.model_path_dict['actdet_deepsort'] = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/actdet_deepsort'
    self.model_path_dict['actdet_acam'] = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/actdet_acam'

  def callSetup(self, model_update_add_list):
    for module_instance in model_update_add_list.split('-'):
      if (module_instance == "exported_mobilenet_v1_1.0_224_preprocess"):
        MobilenetPreprocess.Setup()
      elif (module_instance == "exported_mobilenet_v1_1.0_224_inference"):
        MobilenetInference.Setup()
      elif (module_instance == "tacotron"):
        Tacotron.Setup()
      elif (module_instance == "nlpCPU"):
        Resample.Setup()
      elif (module_instance == "deepspeech2"):
        Deepspeech2.Setup()
      elif (module_instance == "jasper"):
        Jasper.Setup()
      elif (module_instance == "wave2letter"):
        Wave2Letter.Setup()
      elif (module_instance == "encoder"):
        TextEncoder.Setup()
      elif (module_instance == "transformer"):
        Transformer.Setup()
      elif (module_instance == "transformer_big"):
        TransformerBig.Setup()
      elif (module_instance == "conv_s2s"):
        Convs2s.Setup()
      elif (module_instance == "decoder"):
        TextDecoder.Setup()
      elif (module_instance == "actdet_ssd"):
        SSD.Setup()
      elif (module_instance == "actdet_deepsort"):
        DeepSort.Setup()
      elif (module_instance == "actdet_acam"):
        ACAM.Setup()

  def parseRouteTable(self, route_table, route_index):
    tmp = route_table.split("-")
    current_model = tmp[route_index].split(":")[0]
    tt = tmp[route_index + 1].split(":")
    next_model = tt[0]
    next_stub = "%s:%s" % (tt[1], tt[2])
    return current_model, next_model, next_stub

  def addLoadedModelSet(self, model_update_add_list):
    print("[%s][Worker] old loaded_model_set = %s" % (str(time.time()), str(self.loaded_model_set)))
    for module_instance in model_update_add_list.split('-'):
      self.loaded_model_set.add(module_instance)
    print("[%s][Worker] new loaded_model_set = %s" % (str(time.time()), str(self.loaded_model_set)))

  def delLoadedModelSet(self, model_update_del_list):
    print("[%s][Worker] old loaded_model_set = %s" % (str(time.time()), str(self.loaded_model_set)))
    for module_instance in model_update_del_list:
      self.loaded_model_set.remove(module_instance)
    print("[%s][Worker] new loaded_model_set = %s" % (str(time.time()), str(self.loaded_model_set)))

  def getModelConfigStr(self):
    pre = 'model_config_list: {'
    post = '}'
    mid = ''
    for module_instance in self.loaded_model_set:
      if (module_instance in ["nlpCPU", "encoder", "decoder"]): # cpu module -> no need to load in TF-Serving
        continue
      else:
        mid += 'config: {name: "%s", base_path: "%s", model_platform: "tensorflow"}, ' % (module_instance, self.model_path_dict[module_instance])

    return pre + mid + post

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

      elif (current_model == "deepspeech2"):
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

      elif (current_model == "actdet_ssd"):
        module_instance = SSD()

      elif (current_model == "actdet_deepsort"):
        module_instance = DeepSort()

      elif (current_model == "actdet_acam"):
        module_instance = ACAM()

      else:
        print("[Worker] Error...")

      # module_instance.Setup()
      module_instance.PreProcess(request, self.istub)
      module_instance.Apply()
      next_request = module_instance.PostProcess()

      print("[%s][Worker] Received %s result for %s from local TF-Serving, ready for next stub %s\n" % (str(time.time()), current_model, str(frame_info), next_stub))
      next_request.model_spec.name = chain_name
      next_request.model_spec.signature_name = "chain_specification"

      next_request.inputs['frame_info'].CopyFrom(
        tf.make_tensor_proto(str(frame_info)))
      next_request.inputs['route_table'].CopyFrom(
        tf.make_tensor_proto(str(route_table)))
      next_request.inputs['route_index'].CopyFrom(
        tf.make_tensor_proto(route_index + 1, dtype=tf.int32))

      if (next_stub == client_address): # FINAL
        fchannel = grpc.insecure_channel(next_stub)
        fstub = olympian_client_pb2_grpc.OlympianClientStub(fchannel)
        next_result = fstub.Predict(next_request, 30.0)
        fchannel.close()
      else:
        # async way, weird, need to have a time.sleep to guarantee gRPC Predict.future() finished...?
        next_result = self.cstubs[next_stub].Predict.future(next_request, 10.0)
        time.sleep(5)

        # # sync way, might lead to timeout...
        # next_result = self.cstubs[next_stub].Predict(next_request, 30.0)

    elif ("model_update_add" in request.inputs):
      print("========== Update(add) ==========")
      model_update_add_list = str(tensor_util.MakeNdarray(request.inputs["model_update_add"]))
      print("[%s][Worker] Received model_update_add_list = %s" % (str(time.time()), model_update_add_list))

      self.addLoadedModelSet(model_update_add_list)
      self.callSetup(model_update_add_list)
      config_ini = self.getModelConfigStr()
      model_server_config = model_server_config_pb2.ModelServerConfig()
      model_server_config = text_format.Parse(text=config_ini, message=model_server_config)

      internal_request = model_management_pb2.ReloadConfigRequest()
      internal_request.config.CopyFrom(model_server_config)

      internal_response = self.istub_reload.HandleReloadConfigRequest(internal_request, 10.0)

      print("[%s][Worker] Updated model list\n" % str(time.time()))

    elif ("model_update_del" in request.inputs):
      pass

    else: # Not sure yet...
      print("[Worker] Not sure yet...")

    dumbresult = predict_pb2.PredictResponse()
    dumbresult.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))
    return dumbresult    

def main(_):
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
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