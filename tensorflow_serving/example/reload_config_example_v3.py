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
# tf.app.flags.DEFINE_string('model_config_file', '/home/yitao/Documents/edge/edge-serving/reload_model.conf', 'model reconfig file')
FLAGS = tf.app.flags.FLAGS

def getModelConfigStr(model_path_dict, model_update_list):
  pre = 'model_config_list: {'
  post = '}'
  mid = ''
  for module_intance in model_update_list.split('-'):
    mid += 'config: {name: "%s", base_path: "%s", model_platform: "tensorflow"}, ' % (module_intance, model_path_dict[module_intance])

  return pre + mid + post

def main(_):

  model_path_dict = dict()
  model_path_dict['tacotron'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/tacotron'
  model_path_dict['deepspeech2'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/deepspeech2'
  model_path_dict['transformer'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/transformer'
  model_path_dict['jasper'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/jasper'
  model_path_dict['wave2letter'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/wave2letter'
  model_path_dict['conv_s2s'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/conv_s2s'
  model_path_dict['transformer-big'] = '/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/transformer-big'
  model_path_dict['exported_mobilenet_v1_1.0_224_preprocess'] = '/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/exported_mobilenet_v1_1.0_224_preprocess'
  model_path_dict['exported_mobilenet_v1_1.0_224_inference'] = '/home/yitao/Documents/fun-project/tensorflow-related/tensorflow-for-poets-2/exported_mobilenet_v1_1.0_224_inference'

  channel = grpc.insecure_channel(FLAGS.server)
  stub = model_service_pb2_grpc.ModelServiceStub(channel)

  # f = open(FLAGS.model_config_file, "r")
  # config_ini = f.read()
  # config_ini = str('model_config_list: {config: {name: "tacotron",base_path: "/home/yitao/Documents/fun-project/tensorflow-related/miniature-winner/models/tf_servable/tacotron",model_platform: "tensorflow"}}')

  model_update_list = 'tacotron-jasper-deepspeech2'

  config_ini = getModelConfigStr(model_path_dict, model_update_list)

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