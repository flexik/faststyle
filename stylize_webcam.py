"""Use a model to stylize an OpenCV webcam feed."""

import cv2
from multiprocessing import Process, Queue, Array, Value
import tensorflow as tf
from im_transf_net import create_net
import numpy as np
import argparse

def setup_parser():
    """Options for command-line input."""
    parser = argparse.ArgumentParser(description="""Use a trained fast style
                                     transfer model to filter webcam feed.""")
    parser.add_argument("--frame_width", type=int, default=640)
    parser.add_argument("--frame_height", type=int, default=480)
    parser.add_argument('--capture_device', default=0)
    parser.add_argument('--vertical', action='store_true', default=False)
    parser.add_argument('--fullscreen', action='store_true', default=False)
    parser.add_argument('--show_originals', action='store_true', default=False)
    parser.add_argument('--canvas_size', nargs=2, type=int, default=None)
    parser.add_argument('--style_image_path',
                        default='style_images/starry_night_crop.jpg')
    parser.add_argument('--model_path',
                        default='models/starry_final.ckpt',
                        help='Path to .ckpt for the trained model.')
    parser.add_argument('--upsample_method',
                        help="""The upsample method that was used to construct
                        the model being loaded. Note that if the wrong one is
                        chosen an error will be thrown.""",
                        choices=['resize', 'deconv'],
                        default='resize')
    return parser

def capture(last_frame, done, args):
    print "Starting capture..."

    cap = cv2.VideoCapture(args.capture_device)
    #cap.open()
    while not done.value:
        ret, img = cap.read()
        assert ret
        # preprocess frame
        img = cv2.resize(img, (args.frame_width, args.frame_height))
        #img = cv2.flip(img, 1)
        last_frame.raw = img.tostring()

    cap.release()

def processing(last_frame, done, args):
    print "Starting processing..."

    # Set resolution
    x_new, y_new = (args.frame_width, args.frame_height)
    print 'Resolution is: {0} by {1}'.format(x_new, y_new)

    # Create the graph.
    shape = [1, y_new, x_new, 3]
    with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=shape, name='input')
        Y = create_net(X, args.upsample_method)

    # Saver used to restore the model to the session.
    saver = tf.train.Saver()

    if args.vertical:
        t = x_new
        x_new = y_new
        y_new = t

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if args.show_originals:
        style_image = cv2.imread(args.style_image_path)
        style_image = cv2.resize(style_image, (x_new/2, y_new/2))

    # Do not allocate whole GPU memory to Tensorflow.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Begin filtering.
    with tf.Session(config=config) as sess:
        print 'Loading up model...'
        saver.restore(sess, args.model_path)
        print 'Begin filtering...'
        while True:
            # Capture frame-by-frame
            frame = np.frombuffer(last_frame.raw, dtype=np.uint8)
            frame = np.reshape(frame, (args.frame_height, args.frame_width, 3))

            # Make frame 4-D
            img_4d = frame[np.newaxis, :]

            # Our operations on the frame come here
            img_out = sess.run(Y, feed_dict={X: img_4d})
            img_out = np.squeeze(img_out).astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

            # If vertical orientation is used then rotate 90 degrees and mirror
            if args.vertical:
                frame = np.swapaxes(frame, 0, 1)
                img_out = np.swapaxes(img_out, 0, 1)

            # Put the FPS on it.
            # img_out = cv2.putText(img_out, 'fps: {}'.format(fps), (50, 50),
                                  # cv2.FONT_HERSHEY_SIMPLEX,
                                  # 1.0, (255, 0, 0), 3)

            # Display the resulting frame
            if args.show_originals:
                frame = cv2.resize(frame, (x_new/2, y_new/2))
                image_display = np.concatenate((frame, style_image), axis=1)
                output_frame = np.concatenate((img_out, image_display), axis=0)
            else:
                output_frame = img_out

            if args.canvas_size:
                padx = (args.canvas_size[0] - output_frame.shape[1]) // 2
                pady = (args.canvas_size[1] - output_frame.shape[0]) // 2
                output_frame = np.pad(output_frame, ((pady, pady), (padx, padx), (0, 0)), mode='constant')

            cv2.imshow('result', output_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # When everything done, release the capture
    cv2.destroyAllWindows()
    done.value = 1

if __name__ == '__main__':

    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    # Command-line argument parsing.
    parser = setup_parser()
    args = parser.parse_args()

    # set up interprocess communication buffers
    img = 128 * np.ones((args.frame_height, args.frame_width, 3), dtype=np.uint8)
    buf = img.tostring()
    last_frame = Array('c', len(buf))
    last_frame.raw = buf
    done = Value('i', 0)

    # launch capture process
    c = Process(name='capture', target=capture , args=(last_frame, done, args))
    c.start()

    # launch processing process
    p = Process(name='processing', target=processing, args=(last_frame, done, args))
    p.start()

    # wait for processes to finish
    p.join()
    c.join()
