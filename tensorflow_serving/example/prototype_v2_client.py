#!/usr/bin/env python2.7
from __future__ import print_function

import grpc
import time
import numpy as np
from concurrent import futures

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import olympian_master_pb2_grpc
from tensorflow_serving.apis import olympian_worker_pb2_grpc
from tensorflow_serving.apis import olympian_client_pb2_grpc

from tensorflow.python.framework import tensor_util

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 1024 * 1024 * 64

tf.app.flags.DEFINE_string('client', 'localhost:50201', 'Olympian client host:port')
tf.app.flags.DEFINE_string('chain_name', 'mobilenet', 'name of the chain')
FLAGS = tf.app.flags.FLAGS

class OlympianClient(olympian_client_pb2_grpc.OlympianClientServicer):
  
  def load_labels(self):
    label_file = ("/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/retrained_labels.txt")
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

  def Predict(self, request, context):
    if ("FINAL" in request.inputs):
      print("[%s][Client] Received final result w/ peer = %s" % (str(time.time()), context.peer()))
      final_result_value = tensor_util.MakeNdarray(request.inputs["FINAL"])

      # Mobilenet specific
      if (request.model_spec.name == "chain_mobilenet"):
        labels = self.load_labels()
        results = np.squeeze(final_result_value)
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
          print(labels[i], results[i])

    else:
      print("[%s][Client] Something is wrong..." % (str(time.time())))

    # default dumb OK message...
    dumbresult = predict_pb2.PredictResponse()
    dumbresult.outputs["message"].CopyFrom(tf.make_tensor_proto("OK"))
    return dumbresult

def getFirstStub(route_table):
	tmp = route_table.split("-")[0].split(":")
	first_stub = "%s:%s" % (tmp[1], tmp[2])
	return first_stub

def main(_):
  cstubs = dict()

  worker_list = ["localhost:50101", "localhost:50102"]
  for w in worker_list:
  	channel = grpc.insecure_channel(w)
  	stub = olympian_worker_pb2_grpc.OlympianWorkerStub(channel)
  	cstubs[w] = stub

  master_list = ["localhost:50051"]
  for m in master_list:
  	channel = grpc.insecure_channel(m)
  	stub = olympian_master_pb2_grpc.OlympianMasterStub(channel)
  	cstubs[m] = stub

  # start client's sstub
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
  olympian_client_pb2_grpc.add_OlympianClientServicer_to_server(OlympianClient(), server)
  server.add_insecure_port(FLAGS.client)
  server.start()

  # client sends request for sess id and route table
  sess_setup_request = predict_pb2.PredictRequest()
  sess_setup_request.model_spec.name = "chain_mobilenet"
  sess_setup_request.model_spec.signature_name = "chain_specification"
  sess_setup_request.inputs["sess_setup"].CopyFrom(
    tf.make_tensor_proto(FLAGS.client))

  print("[%s][Client] Ready to send sess_setup_request!" % (str(time.time())))
  sess_setup_result = cstubs[master_list[0]].Predict(sess_setup_request, 10.0)
  
  sess_id = str(tensor_util.MakeNdarray(sess_setup_result.outputs["sess_id"]))
  route_table = str(tensor_util.MakeNdarray(sess_setup_result.outputs["route_table"]))

  first_stub = getFirstStub(route_table)
  print("[%s][Client] Received sess_id = %s" % (str(time.time()), sess_id))
  print("                                 first_stub = %s\n" % (first_stub))

  # client sends input requests
  file_list = ["/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg",
               "/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg",
               "/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg",
               "/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg",
               "/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg",
               ]

  for frame_id in range(len(file_list)):
    file_name = file_list[frame_id]
    frame_info = "%s-%s" % (sess_id, frame_id)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = "chain_mobilenet"
    request.model_spec.signature_name = "chain_specification"
    request.inputs["client_input"].CopyFrom(
      tf.make_tensor_proto(file_name))
    request.inputs["frame_info"].CopyFrom(
      tf.make_tensor_proto(frame_info))
    request.inputs["route_table"].CopyFrom(
      tf.make_tensor_proto(route_table))

    print("[%s][Client] Ready to send client_input!" % (str(time.time())))
    result = cstubs[first_stub].Predict(request, 10.0)
    message = tensor_util.MakeNdarray(result.outputs["message"])
    print("[%s][Client] Received message %s\n" % (str(time.time()), message))

  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  tf.app.run()