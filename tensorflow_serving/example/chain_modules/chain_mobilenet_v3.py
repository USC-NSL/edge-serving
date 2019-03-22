import time

import tensorflow as tf
from tensorflow.python.framework import tensor_util

from tensorflow_serving.apis import predict_pb2

class MobilenetPreprocess():
  def Setup(self):
    return

  def PreProcess(self, request, istub):
    request_input = tensor_util.MakeNdarray(request.inputs["client_input"])
    self.istub = istub

    self.internal_request = predict_pb2.PredictRequest()
    self.internal_request.model_spec.name = "exported_mobilenet_v1_1.0_224_preprocess"
    self.internal_request.model_spec.signature_name = 'predict_images'

    self.internal_request.inputs['input_image_name'].CopyFrom(
      tf.make_tensor_proto(request_input))

  def Apply(self):
    self.internal_result = self.istub.Predict(self.internal_request, 10.0)

    # dumb sleep, in order to study sync vs async stub.Predict()
    time.sleep(5)
    
  def PostProcess(self):
    internal_result_value = tensor_util.MakeNdarray(self.internal_result.outputs["normalized_image"])
    next_request = predict_pb2.PredictRequest()
    next_request.inputs['normalized_image'].CopyFrom(
      tf.make_tensor_proto(internal_result_value, shape=[1, 224, 224, 3]))

    return next_request

class MobilenetInference():
  def Setup(self):
    return

  def PreProcess(self, request, istub):
    request_input = tensor_util.MakeNdarray(request.inputs["normalized_image"])
    self.istub = istub

    self.internal_request = predict_pb2.PredictRequest()
    self.internal_request.model_spec.name = "exported_mobilenet_v1_1.0_224_inference"
    self.internal_request.model_spec.signature_name = 'predict_images'
        
    self.internal_request.inputs['normalized_image'].CopyFrom(
      tf.make_tensor_proto(request_input, shape=[1, 224, 224, 3]))

  def Apply(self):
    self.internal_result = self.istub.Predict(self.internal_request, 10.0)

  def PostProcess(self):
    internal_result_value = tensor_util.MakeNdarray(self.internal_result.outputs["scores"])
    next_request = predict_pb2.PredictRequest()
    next_request.inputs['FINAL'].CopyFrom(
      tf.make_tensor_proto(internal_result_value, shape=[1, 5]))

    return next_request