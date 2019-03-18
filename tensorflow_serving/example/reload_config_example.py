# Another way to do this
# https://github.com/sparkingarthur/serving/blob/a59f16e66fe25262a5078dd015f0665f98bf1f83/tensorflow_serving/example/hot_reload_grpc_example.py
# which might have to add new config entries one by one...

from google.protobuf import text_format

import grpc
import tensorflow as tf

from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.config import model_server_config_pb2

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'XXX host:port')
tf.app.flags.DEFINE_string('model_config_file', '/home/yitao/Documents/edge/edge-serving/reload_model.conf', 'model reconfig file')
FLAGS = tf.app.flags.FLAGS

def main(_):

  channel = grpc.insecure_channel(FLAGS.server)
  stub = model_service_pb2_grpc.ModelServiceStub(channel)

  f = open(FLAGS.model_config_file, "r")
  config_ini = f.read()

  model_server_config = model_server_config_pb2.ModelServerConfig()
  model_server_config = text_format.Parse(text=config_ini, message=model_server_config)

  # # if want to add mnist to FLAGS.model_config_file list
  # new_config = model_server_config.model_config_list.config.add()
  # new_config.name = "mnist"
  # new_config.base_path = "/your/path/to/mnist_model"
  # new_config.model_platform = "tensorflow"

  request = model_management_pb2.ReloadConfigRequest()
  request.config.CopyFrom(model_server_config)

  print(request.IsInitialized())
  print(request.ListFields())

  responese = stub.HandleReloadConfigRequest(request, 10.0)

  if responese.status.error_code == 0:
      print("Reload sucessfully")
  else:
      print("Reload failed!")
      print(responese.status.error_code)
      print(responese.status.error_message)


if __name__ == '__main__':
  tf.app.run()