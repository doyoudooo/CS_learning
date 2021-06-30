import tensorflow as tf
print("GPU",tf.test.is_gpu_available())
a=tf.constant(3.)
b=tf.constant(6.)
print(a*b)