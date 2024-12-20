import tensorflow as tf

# Test TensorFlow operations on GPU
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([4.0, 5.0, 6.0], shape=[3], name='b')
    c = a + b
    print(c)


#to try it: python3 test-tensorflow.py