# image warping and stereo rectification in opencv and tensorflow

import tensorflow as tf
import numpy as np
import cv2 as cv

width = 64
height = 64

def _warp_matrix(bbox, bbox_2, Hr):
    """
        Calculate the true warping matrix including the crop and resize of the input img to the output heatmap
        Because the Hr matrix is defined on the source img
        ref: Monet: multiview semi-supervised keypoint via epipolar divergence
    """

    s = 64 / tf.maximum(bbox[2], bbox[3]) #tf.cast(256 / tf.maximum(bbox[2], bbox[3]), 'float32') # input network cropped image size / source cropped image size
    Hb = tf.convert_to_tensor([[s, 0, -s*bbox[0]],
                                [0, s, -s*bbox[1]],
                                [0, 0, 1]])
    Hb_inv = tf.cast(tf.matrix_inverse(Hb), 'float32')
    s_2 = 64 / tf.maximum(bbox_2[2], bbox_2[3])
    Hb_hat = tf.convert_to_tensor([[s_2, 0, -s_2*bbox_2[0]],
                                    [0, s_2, -s_2*bbox_2[1]],
                                    [0, 0, 1.0]])
    Hb_hat = tf.cast(Hb_hat, 'float32')

    return tf.matmul(tf.matmul(Hb_hat,tf.cast(tf.convert_to_tensor(Hr), 'float32')), Hb_inv)
    
    # s_h = 64 / 256 # output heatmap / input network cropped image
    # Hc = tf.convert_to_tensor([[s_h, 0, 0],
    #                             [0, s_h, 0],
    #                             [0, 0, 1]])
    # Hc_inv = tf.cast(tf.matrix_inverse(Hc), 'float32')
    # s = 256 / tf.maximum(bbox[2], bbox[3]) #tf.cast(256 / tf.maximum(bbox[2], bbox[3]), 'float32') # input network cropped image size / source cropped image size
    # Hb = tf.convert_to_tensor([[s, 0, -s*bbox[0]],
    #                             [0, s, -s*bbox[1]],
    #                             [0, 0, 1]])
    # Hb_inv = tf.cast(tf.matrix_inverse(Hb), 'float32')
    # Hb_hat = tf.convert_to_tensor([[s, 0, -s*bbox[0]],
    #                                 [0, s, -s*bbox[1]],
    #                                 [0, 0, 1.0]])
    # Hb_hat = tf.cast(Hb_hat, 'float32')
    # Hc_hat = tf.convert_to_tensor([[s_h, 0, 0],
    #                                 [0, s_h, 0],
    #                                 [0, 0, 1.0]])
    # Hc_hat = tf.cast(Hc_hat, 'float32')
    # return tf.matmul(tf.matmul(tf.matmul(Hc_hat, Hb_hat),tf.cast(tf.convert_to_tensor(Hr), 'float32')), tf.matmul(Hb_inv, Hc_inv))

def meshgrid():
    with tf.variable_scope('meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(0.0, width-1, width), 1), [1, 0])) # 64*1 * 1*64
        y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, height-1, height), 1),
                        tf.ones(shape=tf.stack([1, width]))) # 64*1 * 1*64
        # y_t: [1...1]      x_t: [1 ..0.. -1]
        #      .0...0.            . ..0.. .
        #      [-1...-1]		 [1 ..0.. -1]
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
        return grid

def _interpolate(image_src, x, y):
    '''
        # Bilinear interpolation by grid methods
    '''
    with tf.variable_scope('_interpolate'):
        # constants
        height_f = tf.cast(tf.shape(image_src)[0], 'float32')
        width_f = tf.cast(tf.shape(image_src)[1], 'float32')
        width = tf.cast(width_f, 'int32')
        height = tf.cast(height_f, 'int32')
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        
        # bilinear sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        
        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        # calculate interpolated values with weights
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        
        # Use 1-D index instead of 2-D index, so take the width into consideration
        base_y0 = y0 * width
        base_y1 = y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        # use indices to lookup pixels in the flat image and restore
        im_flat = tf.cast(tf.reshape(image_src, [-1]), 'float32')
        Ia = tf.expand_dims(tf.gather(im_flat, idx_a), 1)
        Ib = tf.expand_dims(tf.gather(im_flat, idx_b), 1)
        Ic = tf.expand_dims(tf.gather(im_flat, idx_c), 1)
        Id = tf.expand_dims(tf.gather(im_flat, idx_d), 1)

        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

def warp_image(image_src, bbox, bbox_2, Hr):
    with tf.name_scope('Warp_image'):
    
        # test_matrix = np.eye(3)
        # test_matrix[0,2] = 10
        # test_matrix = tf.convert_to_tensor(test_matrix)
        # Test_matrix_inv = tf.cast(tf.matrix_inverse(test_matrix), 'float32')

        warp_matrix = _warp_matrix(bbox, bbox_2, Hr)
        warp_matrix_inv = tf.cast(tf.matrix_inverse(warp_matrix), 'float32')

        # grid of (x_t, y_t, 1) (target) in ref
        grid = meshgrid()
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(warp_matrix_inv, grid) 
        T_g = T_g / T_g[2]
        x_s_flat = T_g[0,:]
        y_s_flat = T_g[1,:]
        heatmap_warp_r = tf.expand_dims(tf.reshape(_interpolate(image_src[:,:,0], x_s_flat, y_s_flat), tf.stack([height, width])), -1)
        heatmap_warp_g = tf.expand_dims(tf.reshape(_interpolate(image_src[:,:,1], x_s_flat, y_s_flat), tf.stack([height, width])), -1)
        heatmap_warp_b = tf.expand_dims(tf.reshape(_interpolate(image_src[:,:,2], x_s_flat, y_s_flat), tf.stack([height, width])), -1)
        heatmap_warp = tf.concat([heatmap_warp_r,heatmap_warp_g,heatmap_warp_b], 2)
        return heatmap_warp, warp_matrix

if __name__ == "__main__":
    
    bbox_00 = [935, 480, 1367, 988] # cam_00_0.2_30_001
    bbox_01 = [1100, 832, 1640, 1364] #cam_01_0.2_30_001

    Hr_00_01 = tf.convert_to_tensor([[ 4.32883423e-01, -4.15206530e-01,  1.54938615e+03],
       [ 1.51482006e-01,  9.36391187e-01,  1.78045473e+01],
       [-3.13256900e-04,  6.45963375e-05,  1.12876506e+00]])
    Hr_01_00 = tf.convert_to_tensor([[ 9.96265636e-01,  6.27777492e-01, -1.26918249e+03],
       [-3.06876808e-01,  9.26311729e-01,  4.36239218e+02],
       [ 2.35777033e-04,  9.16596328e-05,  5.93788589e-01]])
    
    Hr_00_01_np = np.asarray([[ 4.32883423e-01, -4.15206530e-01,  1.54938615e+03],
       [ 1.51482006e-01,  9.36391187e-01,  1.78045473e+01],
       [-3.13256900e-04,  6.45963375e-05,  1.12876506e+00]])
    Hr_01_00_np = np.asarray([[ 9.96265636e-01,  6.27777492e-01, -1.26918249e+03],
       [-3.06876808e-01,  9.26311729e-01,  4.36239218e+02],
       [ 2.35777033e-04,  9.16596328e-05,  5.93788589e-01]])

    bbox_00_new_view = []
    for i in range(2):
        for j in range(2):
            corner = [bbox_00[i*2], bbox_00[2*j+1],1]
            corner_new = np.dot(Hr_00_01_np, np.asarray(corner).T)
            corner_new = corner_new / corner_new[2]
            bbox_00_new_view.append(corner_new[:2])
    bbox_00_new_view = np.asarray(bbox_00_new_view)
    bbox_01_new_view = []
    for i in range(2):
        for j in range(2):
            corner = [bbox_01[i*2], bbox_01[2*j+1],1]
            corner_new = np.dot(Hr_01_00_np, np.asarray(corner).T)
            corner_new = corner_new / corner_new[2]
            bbox_01_new_view.append(corner_new[:2])
    bbox_01_new_view = np.asarray(bbox_01_new_view)
    bbox_00_new_view =[np.min(bbox_00_new_view[:,0]), np.min(bbox_00_new_view[:,1]), np.max(bbox_00_new_view[:,0]), np.max(bbox_00_new_view[:,1])]
    bbox_01_new_view =[np.min(bbox_01_new_view[:,0]), np.min(bbox_01_new_view[:,1]), np.max(bbox_01_new_view[:,0]), np.max(bbox_01_new_view[:,1])]

    bbox_00 = [bbox_00[0], bbox_00[1], bbox_00[2] - bbox_00[0], bbox_00[3] - bbox_00[1]]
    bbox_01 = [bbox_01[0], bbox_01[1], bbox_01[2] - bbox_01[0], bbox_01[3] - bbox_01[1]]
    bbox_00_new_view = [bbox_00_new_view[0], bbox_00_new_view[1], bbox_00_new_view[2] - bbox_00_new_view[0], bbox_00_new_view[3] - bbox_00_new_view[1]]
    bbox_01_new_view = [bbox_01_new_view[0], bbox_01_new_view[1], bbox_01_new_view[2] - bbox_01_new_view[0], bbox_01_new_view[3] - bbox_01_new_view[1]]
    
    image_view_00_run = cv.imread('./cam_00_0.2_30_001.png')
    image_view_01_run = cv.imread('./cam_01_0.2_30_001.png')

    # first crop then resize to 256 then resize to 64
    max_l = max(bbox_00[2], bbox_00[3]) # choose max in width and height for sure that it's a square.
    crop_size = max_l
    image_view_00_crop_run = image_view_00_run[bbox_00[1] : bbox_00[1]+crop_size, bbox_00[0] : bbox_00[0]+crop_size]
    max_l = max(bbox_01[2], bbox_01[3]) # choose max in width and height for sure that it's a square.
    crop_size = max_l
    image_view_01_crop_run = image_view_01_run[bbox_01[1] : bbox_01[1]+crop_size, bbox_01[0] : bbox_01[0]+crop_size]
    image_view_00_run = cv.resize(image_view_00_crop_run, (width, height), interpolation=cv.INTER_CUBIC)
    image_view_01_run = cv.resize(image_view_01_crop_run, (width, height), interpolation=cv.INTER_CUBIC)
    cv.imwrite('./cam_00_0.2_30_001_crop.png', image_view_00_run)
    cv.imwrite('./cam_01_0.2_30_001_crop.png', image_view_01_run)

    with tf.name_scope('Warp_image_test'):
        image_view_00 = tf.placeholder(dtype = tf.float32, shape= (width, height, 3), name = 'input_img_00')
        image_view_01 = tf.placeholder(dtype = tf.float32, shape= (width, height, 3), name = 'input_img_01')
        image00_warped_to_01, _ = warp_image(image_view_00, bbox_00, bbox_00_new_view, Hr_00_01)
        image01_warped_to_00, _ = warp_image(image_view_01, bbox_01, bbox_01_new_view, Hr_01_00)
        with tf.Session() as sess:
            image00_warped_to_01_save, image01_warped_to_00_save = \
                sess.run([image00_warped_to_01, image01_warped_to_00], feed_dict = {image_view_00:image_view_00_run, image_view_01:image_view_01_run})
            print('Warp Done')
            cv.imwrite('./image00_warped.png', image00_warped_to_01_save)
            cv.imwrite('./image01_warped.png', image01_warped_to_00_save)


# warping test with opencv
# camera1_matrix = np.asarray([[1502.76649, 0.0, 1010.64134],
#                             [0.0, 1502.52625, 1002.16243],
#                             [0.0, 0.0, 1.0]])
# dist1 = np.zeros((1,5))
# camera2_matrix = np.asarray([[ 1506.46911, 0.0, 1008.10591],
#                             [0.0, 1506.47502, 1015.30310],
#                             [0.0, 0.0, 1.0]])
# dist2 = np.zeros((1,5))
# camera_average = (camera1_matrix + camera2_matrix)/2
# # R = np.dot(R[camera_2], np.linalg.inv(R[camera_1]))
# R = np.asarray([[ 0.14704824, -0.80610698,  0.57320882],
#                 [ 0.72411806,  0.48250588,  0.49278911],
#                 [-0.67381737,  0.34260709,  0.65466827]])

# # T = T[camera_2] - T[camera_1]
# T = np.asarray([-0.22870329,
#                 -0.16141065,
#                 0.11197078])

# R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(camera_average, dist1, camera_average,dist2,image_view_00_run.shape[:2], R, T, alpha = 1.0)
# mapx1, mapy1 = cv.initUndistortRectifyMap(camera_average, dist1, R1, camera_average, image_view_00_run.shape[:2], cv.CV_32F)
# mapx2, mapy2 = cv.initUndistortRectifyMap(camera_average, dist2, R2, camera_average, image_view_00_run.shape[:2], cv.CV_32F)
# img_rect1 = cv.remap(image_view_00_run, mapx1, mapy1, cv.INTER_LINEAR)
# img_rect2 = cv.remap(image_view_01_run, mapx2, mapy2, cv.INTER_LINEAR)
# cv.imwrite('Accurate_00.png', img_rect1)
# cv.imwrite('Accurate_01.png', img_rect2)
# R1 R2
# [[ 0.74947379 -0.48041345  0.45551284]
# [ 0.46549072  0.87165516  0.15341338]
# [-0.47075197  0.09705769  0.8769107 ]]
# [[ 0.75857742  0.53537697 -0.37139171]
# [-0.54625981  0.83324942  0.08541438]
# [ 0.35519082  0.13808295  0.92453912]]