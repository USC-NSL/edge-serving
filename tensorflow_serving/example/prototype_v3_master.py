#!/usr/bin/env python2.7
from __future__ import print_function

import grpc
import time
from concurrent import futures
import urllib2
import operator
import threading

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.framework import tensor_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import olympian_master_pb2_grpc
from tensorflow_serving.apis import olympian_worker_pb2_grpc
from tensorflow_serving.apis import olympian_client_pb2_grpc

import logging
logging.basicConfig()

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64
MAX_WORKERS = 60

# # Worker's loaded model set class
# class WorkerLoadedModelSet():
#   def __init__(self, w_address):
#     self.w_address = w_address
#     self.loaded_model_set = set()

# Master Class
class OlympianMaster(olympian_master_pb2_grpc.OlympianMasterServicer):

  # def retrospect(self):
  #   # while(True):
  #   if (True): # one time route table update...
  #     time.sleep(10)
  #     for sess_id in self.registered_session:
  #       tmp = sess_id.split('-')
  #       chain_name = tmp[0]
  #       client_address = tmp[1]
  #       if (chain_name == "chain_mobilenet"):
  #         new_route_table = "hahahahahahahahahahahahahahahahahahahaha"
  #         retrospect_request = predict_pb2.PredictRequest()
  #         retrospect_request.model_spec.name = "unknown"
  #         retrospect_request.model_spec.signature_name = 'unknown'

  #         retrospect_request.inputs['update_route_table'].CopyFrom(
  #           tf.make_tensor_proto(new_route_table))

  #         fchannel = grpc.insecure_channel(client_address)
  #         fstub = olympian_client_pb2_grpc.OlympianClientStub(fchannel)

  #         retrospect_result = fstub.Predict(retrospect_request, 10.0)

  #       else:
  #         print("Not implemented yet...")

  def __init__(self):
    self.cstubs = dict()

    # # self.worker_list = ["localhost:50101", "localhost:50102"]
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

    self.registered_session = []

    self.registered_worker = dict()
    for w in self.worker_list:
      # self.registered_worker[w] = WorkerLoadedModelSet(w)
      self.registered_worker[w] = set()

    # t = threading.Thread(target = self.retrospect)
    # t.start()

  def getRemainingResource(self, p_address):
    responses = urllib2.urlopen('http://%s/monitoring/prometheus/metrics' % p_address).readlines()
    memory_max = -1
    memory_current = -1

    for response in responses:
      line = response.rstrip()
      if (line[0] == '#'):
        continue
      else:
        key = line.split('{')[0]
        value = line.split('}')[1]
        if (key == ':tensorflow:core:common_runtime:bfc_allocator_memory_limit'):
          memory_max = int(value)
        elif (key == ':tensorflow:core:common_runtime:bfc_allocator_memory'):
          memory_current = int(value)

    memory_remain = memory_max - memory_current
    # print("memory_max = %d" % memory_max)
    # print("memory_current = %d" % memory_current)
    return memory_remain

  def getRouteTable_helper(self, chain_instance, chain_profile, resource_map):
    total_profile = sum(chain_profile)
    for w_address, remaining_resource in sorted(resource_map.items(), key=operator.itemgetter(1), reverse = True):
      if (remaining_resource >= total_profile):
        base_route_table = ""
        for module_intance in chain_instance:
          base_route_table += "%s:%s-" % (module_intance, w_address)
        return base_route_table
      else:
        # Not implemented yet...
        return "Not implemented yet..."

  def getRouteTable(self, chain_name, client_address, sess_requirement):
    prometheus_list = []
    for w in self.worker_list:
      tmp = w.split(":")
      # If one machine has two GPUs, this machine will run two workers, each with one TF-Serving
      # And these two workers, each will have its ip:port in worker_list, so we need two p_address for them.
      p_address = "%s:%d" % (tmp[0], int(tmp[1]) + 5000)
      prometheus_list.append(p_address)

    resource_map = dict()
    for p in prometheus_list:
      remaining_resource = self.getRemainingResource(p)
      tmp = p.split(":")
      w_address = "%s:%s" % (tmp[0], int(tmp[1]) - 5000)
      resource_map[w_address] = remaining_resource

    if (chain_name == "chain_mobilenet"):
      default_chain_instance = ["exported_mobilenet_v1_1.0_224_preprocess", "exported_mobilenet_v1_1.0_224_inference"]
      default_chain_profile = [1000, 5000]
      base_route_table = self.getRouteTable_helper(default_chain_instance, default_chain_profile, resource_map)
      route_table = "%sFINAL:%s" % (base_route_table, client_address)
      return route_table
    elif (chain_name == "chain_nlp_speech"):
      default_chain_instance = ["tacotron", "nlpCPU", "deepspeech2", "encoder", "transformer_big", "decoder"]
      # default_chain_instance = ["tacotron", "nlpCPU", "jasper", "encoder", "transformer_big", "decoder"]
      # default_chain_instance = ["tacotron", "nlpCPU", "wave2letter", "encoder", "transformer_big", "decoder"]
      default_chain_profile = [1000, 1000, 1000, 1000, 1000, 1000, 1000]
      base_route_table = self.getRouteTable_helper(default_chain_instance, default_chain_profile, resource_map)
      route_table = "%sFINAL:%s" % (base_route_table, client_address)
      return route_table
    elif (chain_name == "chain_nlp_transform"):
      default_chain_instance = ["tacotron", "nlpCPU", "jasper", "encoder", "transformer", "decoder"]
      # default_chain_instance = ["tacotron", "nlpCPU", "jasper", "encoder", "transformer_big", "decoder"]
      # default_chain_instance = ["tacotron", "nlpCPU", "jasper", "encoder", "conv_s2s", "decoder"]
      default_chain_profile = [1000, 1000, 1000, 1000, 1000, 1000, 1000]
      base_route_table = self.getRouteTable_helper(default_chain_instance, default_chain_profile, resource_map)
      route_table = "%sFINAL:%s" % (base_route_table, client_address)
      return route_table
    else:
      return "Error, something is wrong..."

  def add_worker_loaded_model(self, route_table):
    myWorkerModelMap = dict()
    for tmp in route_table.split('-'):
      tt = tmp.split(':')
      module_intance = tt[0]
      w_address = "%s:%s" % (tt[1], tt[2])
      # if (module_intance in ["FINAL", "nlpCPU", "encoder", "decoder"]):
      if (module_intance in ["FINAL"]):
        continue
      else:
        if (module_intance in self.registered_worker[w_address]):
          continue
        else:
          self.registered_worker[w_address].add(module_intance)
          if (w_address not in myWorkerModelMap):
            myWorkerModelMap[w_address] = ""
          myWorkerModelMap[w_address] += "%s-" % module_intance

    for w_address, model_update_list_tmp in myWorkerModelMap.iteritems():
      model_update_list = model_update_list_tmp[:-1]
      print("[%s][Master] Notifying %s to add %s" % (str(time.time()), w_address, model_update_list))

      model_update_request = predict_pb2.PredictRequest()
      model_update_request.model_spec.name = "unknown"
      model_update_request.model_spec.signature_name = "unknown"
      model_update_request.inputs["model_update_add"].CopyFrom(
        tf.make_tensor_proto(model_update_list))

      model_update_result = self.cstubs[w_address].Predict(model_update_request, 10.0)

  def printWorkerModelInfo(self):
    print("[%s][Master] workers' model info is:" % str(time.time()))
    for w_address, loaded_model_set in self.registered_worker.iteritems():
      print("%s-%s" % (w_address, str(loaded_model_set)))

    print(' ')

  def Predict(self, request, context):
    print("========== Predict() ==========")
    if ("sess_setup" in request.inputs): # gRPC of sess id request from client
      chain_name = str(request.model_spec.name)
      client_address = str(tensor_util.MakeNdarray(request.inputs["sess_setup"]))
      sess_requirement = str(tensor_util.MakeNdarray(request.inputs["sess_requirement"]))

      print("[%s][Master] Received sess id request for %s w/ client_address = %s, requirement = %s" % (str(time.time()), chain_name, client_address, sess_requirement))
      route_table = self.getRouteTable(chain_name, client_address, sess_requirement)
      print("[%s][Master] Generated default route table: %s" % (str(time.time()), route_table))

      self.add_worker_loaded_model(route_table)
      print("[%s][Master] Worker has loaded the required models" % str(time.time()))
      self.printWorkerModelInfo()

      sess_id = "%s-%s" % (chain_name, client_address)
      self.registered_session.append(sess_id)

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
  
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
  olympian_master_pb2_grpc.add_OlympianMasterServicer_to_server(OlympianMaster(), server)
  server.add_insecure_port('localhost:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  tf.app.run()