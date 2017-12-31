# Behavior Clone

The following two packages need to be installed to run `drive.py`.

```bash
pip install python-socketio eventlet pillow flask h5py
```

There is some versions issue among Python, Keras, and Tensorflow. Keras and TensorFlow need to be updated to the latest version, otherwise there may be "unknown opcode" or "bad marshal data" errors when a lambda layer is used to normalize the images. And this seems to only work for Python 3.5 -- once I update Python to 3.6 the problem comes back even if both Keras and TensorFlow were updated.

```bash
pip install tensorflow --upgrade
pip install keras --upgrade
```
