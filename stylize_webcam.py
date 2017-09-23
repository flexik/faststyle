"""Use a model to stylize an OpenCV webcam feed.

File author: Grant Watson
Date: Feb 2017
"""

import cv2
import tensorflow as tf
from im_transf_net import create_net
import numpy as np
import argparse

def setup_parser():
    """Options for command-line input."""
    parser = argparse.ArgumentParser(description="""Use a trained fast style
                                     transfer model to filter webcam feed.""")
    parser.add_argument('--capture_device', default=1)
    parser.add_argument('--vertical', action='store_true', default=False)
    parser.add_argument('--fullscreen', action='store_true', default=False)
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
    parser.add_argument('--resolution',
                        help="""Dimensions for webcam. Note that, depending on
                        the webcam, only certain resolutions will be possible.
                        Leave this argument blank if want to use default
                        resolution.""",
                        nargs=2,
                        type=int,
                        default=None)
    return parser


if __name__ == '__main__':

    # Command-line argument parsing.
    parser = setup_parser()
    args = parser.parse_args()
    model_path = args.model_path
    upsample_method = args.upsample_method
    resolution = args.resolution

    # Instantiate video capture object.
    cap = cv2.VideoCapture(args.capture_device)

    # Set resolution
    if resolution is not None:
        x_length, y_length = resolution
        cap.set(3, x_length)  # 3 and 4 are OpenCV property IDs.
        cap.set(4, y_length)
    x_new = int(cap.get(3))
    y_new = int(cap.get(4))
    print 'Resolution is: {0} by {1}'.format(x_new, y_new)

    # Create the graph.
    shape = [1, y_new, x_new, 3]
    with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=shape, name='input')
        Y = create_net(X, upsample_method)

    # Saver used to restore the model to the session.
    saver = tf.train.Saver()

    if args.vertical:
        t = x_new
        x_new = y_new
        y_new = t

    if args.fullscreen:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    style_image = cv2.imread(args.style_image_path)
    style_image = cv2.resize(style_image, (x_new/2, y_new/2))
    
    # Begin filtering.
    with tf.Session() as sess:
        print 'Loading up model...'
        saver.restore(sess, model_path)
        print 'Begin filtering...'
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            assert ret

            # Make frame 4-D
            img_4d = frame[np.newaxis, :]

            # Our operations on the frame come here
            img_out = sess.run(Y, feed_dict={X: img_4d})
            img_out = np.squeeze(img_out).astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

            # If vertical orientation is used then rotate 90 degrees
            if args.vertical:
                frame = np.fliplr(np.swapaxes(frame, 0, 1))
                img_out = np.fliplr(np.swapaxes(img_out, 0, 1))

            # Put the FPS on it.
            # img_out = cv2.putText(img_out, 'fps: {}'.format(fps), (50, 50),
                                  # cv2.FONT_HERSHEY_SIMPLEX,
                                  # 1.0, (255, 0, 0), 3)
            
            # Display the resulting frame
            frame = cv2.resize(frame, (x_new/2, y_new/2))
            image_display = np.concatenate((frame, style_image), axis=1)
            output_frame = np.concatenate((img_out, image_display), axis=0)
            
            if args.canvas_size:
                padx = (args.canvas_size[0] - output_frame.shape[1]) // 2
                pady = (args.canvas_size[1] - output_frame.shape[0]) // 2
                output_frame = np.pad(output_frame, ((pady, pady), (padx, padx), (0, 0)), mode='constant')
            
            cv2.imshow('result', output_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
