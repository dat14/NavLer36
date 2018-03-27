from network import *
from data_process import *
import tensorflow as tf


def mainci():

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

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Training Part

    # Load the data
        sess.run(init)
        saver.restore(sess, path + 'PoseNet.ckpt')
        data_gen = gen_data_batch_val(validation_datasource)

        for i in range(len(validation_datasource.validation_images)):
            for j in range(10):

                np_images, pose_val = next(data_gen)
                feed = {images: np_images}

                predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)

                for i in range(32):

                    distance_xs = []
                    heights = []
                    sways = []
                    pitchs = []
                    turns = []

                    distance_x= predicted_x[i,0] / pose_val[i,0]
                    distance_xs = np.append(distance_xs, distance_x)
                    height = predicted_x[i,1] / pose_val[i,1]
                    heights = np.append(heights, height)
                    sway= predicted_q[i,0] / pose_val[i,2]
                    sways = np.append(sways, sway)
                    pitch = predicted_q[i,1] / pose_val[i,3]
                    pitchs = np.append(pitchs, pitch)
                    turn = predicted_q[i,2] / pose_val[i,4]
                    turns = np.append(turns, turn)

                distance_xs = np.sum(distance_xs, axis=0)
                heights = np.sum(heights, axis=0)
                sways = np.sum(sways, axis=0)
                pitchs = np.sum(pitchs, axis=0)
                turns = np.sum(turns, axis=0)

                print("Iteration %d" % j)
                print(distance_xs)
                print(heights)
                print(sways)
                print(pitchs)
                print(turns)

                distance_xss = []
                heightss = []
                swayss = []
                pitchss = []
                turnss = []

                distance_xss = np.append(distance_xss, distance_xs)
                heightss = np.append(heightss, heights)
                swayss = np.append(swayss, sways)
                pitchss = np.append(pitchss, pitchs)
                turnss = np.append(turnss, turns)

                distance_xss = np.sum(distance_xss, axis=0)
                heightss = np.sum(heightss, axis=0)
                swayss = np.sum(swayss, axis=0)
                pitchss = np.sum(pitchss, axis=0)
                turnss = np.sum(turnss, axis=0)

        print("Average")
        print(distance_xss)
        print(heightss)
        print(swayss)
        print(pitchss)
        print(turnss)


if __name__ == '__mainci__':
    mainci()