from network import *
from data_process import *
import tensorflow as tf


def main():
    # Measuring program snipet execution time.
    start_time = time.time()

    images = tf.placeholder(tf.float32, [batch_size, 240, 320, 3])
    images_val = tf.placeholder(tf.float32, [batch_size, 240, 320, 3])
    poses_x = tf.placeholder(tf.float32, [batch_size, 2])
    poses_q = tf.placeholder(tf.float32, [batch_size, 3])
    #	pose_val = tf.placeholder(tf.float32, [batch_size, 3])
    datasource = get_data()
    #	a, b, c = get_data1()
    #	datasource = get_data2(a, b, c)
    #	datasource = get_data1()
    train_datasource, validation_datasource, test_datasource = train_validate_test_split(datasource)

    net = GoogLeNet({'data': images})

    p1_x = net.layers['cls1_fc_pose_xyz']
    p1_q = net.layers['cls1_fc_pose_wpqr']
    p2_x = net.layers['cls2_fc_pose_xyz']
    p2_q = net.layers['cls2_fc_pose_wpqr']
    p3_x = net.layers['cls3_fc_pose_xyz']
    p3_q = net.layers['cls3_fc_pose_wpqr']

    l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x))))  # * 0.3
    l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q))))  # * 150
    l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x))))  # * 0.3
    l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q))))  # * 150
    l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x))))  # * 1
    l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q))))  # * 500

    loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q
    opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False,
                                 name='Adam').minimize(loss)

    # Set GPU options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    outputFile = "C:/Users/trand/Desktop/PseudoCorridor/Merged/PoseNet.ckpt"

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Training Part

        # Load the data
        sess.run(init)
        saver.restore(sess, path + 'PoseNet.ckpt')
        # net.load('/home/duyan/Downloads/PseudoCorridor/Pass1/input_data/posenet.npy', sess)
        data_gen = gen_data_batch(train_datasource)
        for i in range(max_iterations):
            np_images, np_poses_x, np_poses_q = next(data_gen)
            feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}

            sess.run(opt, feed_dict=feed)
            np_loss = sess.run(loss, feed_dict=feed)
            if i % 20 == 0:
                print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(np_loss))
            if i % 60 == 0:
                #saver.save(sess, outputFile)
                print("Intermediate file saved at: " + outputFile)
        #saver.save(sess, outputFile)
        print("Intermediate file saved at: " + outputFile)
    print("\n--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()