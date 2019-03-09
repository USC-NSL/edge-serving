sudo rm /usr/local/bin/tensorflow_model_server
sudo pip uninstall -y tensorflow-serving-api-gpu
bazel build -c opt --config=cuda --config=nativeopt --copt="-fPIC" tensorflow_serving/...
sudo cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/

bazel build -c opt --config=cuda --config=nativeopt tensorflow_serving/tools/pip_package:build_pip_package
sudo bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package /tmp/tensorflow_serving_package
sudo pip install /tmp/tensorflow_serving_package/tensorflow_serving_api_gpu-1.13.0-py2.py3-none-any.whl
