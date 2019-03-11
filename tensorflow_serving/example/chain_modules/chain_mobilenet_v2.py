import tensorflow as tf
from tensorflow.python.framework import tensor_util

from tensorflow_serving.apis import predict_pb2

import time

class MobilenetPreprocess():
  def Setup(self):
    return

  def PreProcess(self, request, route_table, current_model, next_stub, istub, cstub):
    self.chain_name = request.model_spec.name
    self.route_table = route_table
    self.current_model = current_model
    self.next_stub = next_stub
    self.request_input = tensor_util.MakeNdarray(request.inputs["client_input"])
    self.istub = istub
    self.cstub = cstub

    print("========== Predict() ==========")
    print("[%s][Worker] Received request using chain %s w/ request_input.shape = %s" % (str(time.time()), self.chain_name, str(self.request_input.shape)))
    print("[%s][Worker] current_model = %s" % (time.time(), self.current_model))
    print("                        next_stub = %s" % (self.next_stub))

    self.internal_request = predict_pb2.PredictRequest()
    self.internal_request.model_spec.name = "exported_mobilenet_v1_1.0_224_preprocess"
    self.internal_request.model_spec.signature_name = 'predict_images'

    self.internal_request.inputs['input_image_name'].CopyFrom(
      tf.make_tensor_proto(self.request_input))

    return

  def Apply(self):
    self.internal_result = self.istub.Predict(self.internal_request, 10.0)
    return

  def PostProcess(self):
    internal_result_value = tensor_util.MakeNdarray(self.internal_result.outputs["normalized_image"])
    print("[%s][Worker] Received internal result, ready for next_stub %s\n" % (str(time.time()), self.next_stub))
        
    next_request = predict_pb2.PredictRequest()
    next_request.model_spec.name = self.chain_name
    next_request.model_spec.signature_name = "chain_specification"

    next_request.inputs['normalized_image'].CopyFrom(
      tf.make_tensor_proto(internal_result_value, shape=[1, 224, 224, 3]))
    next_request.inputs['route_table'].CopyFrom(
      tf.make_tensor_proto(str(self.route_table)))

    next_result = self.cstub.Predict(next_request, 10.0)

    return

class MobilenetInference():
  def Setup(self):
    return

  def PreProcess(self, request, route_table, current_model, next_stub, istub, cstub):
    self.chain_name = request.model_spec.name
    self.route_table = route_table
    self.current_model = current_model
    self.next_stub = next_stub
    self.request_input = tensor_util.MakeNdarray(request.inputs["normalized_image"])
    self.istub = istub
    self.cstub = cstub

    print("========== Predict() ==========")
    print("[%s][Worker] Received request using chain %s w/ request_input.shape = %s" % (str(time.time()), self.chain_name, str(self.request_input.shape)))
    print("[%s][Worker] current_model = %s" % (time.time(), self.current_model))
    print("                        next_stub = %s" % (self.next_stub))

    self.internal_request = predict_pb2.PredictRequest()
    self.internal_request.model_spec.name = "exported_mobilenet_v1_1.0_224_inference"
    self.internal_request.model_spec.signature_name = 'predict_images'
        
    self.internal_request.inputs['normalized_image'].CopyFrom(
      tf.make_tensor_proto(self.request_input, shape=[1, 224, 224, 3]))

    return

  def Apply(self):
    self.internal_result = self.istub.Predict(self.internal_request, 10.0)
    return

  def PostProcess(self):
    internal_result_value = tensor_util.MakeNdarray(self.internal_result.outputs["scores"])
    print("[%s][Worker] Received internal result, ready for next_stub %s\n" % (str(time.time()), self.next_stub))
    next_request = predict_pb2.PredictRequest()
    next_request.model_spec.name = self.chain_name
    next_request.model_spec.signature_name = "chain_specification"

    next_request.inputs['FINAL'].CopyFrom(
      tf.make_tensor_proto(internal_result_value, shape=[1, 5]))

    next_result = self.cstub.Predict(next_request, 10.0)
        
    return
