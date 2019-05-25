
import imageio
import os

img_directory = r'/home/eason/Mouse_behavior/2D_Model/testset/results_not_train'
def create_gif(image_list, gif_name):
 
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(img_directory, image_name)))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.2)
 
    return

def gather_image_list(cam_name, image_list_full):
    image_list = []
    # for image in image_list_full:
    #     if image[0:13] == cam_name:
    #         image_list.append(image)
    for i in range(30):
        image_list.append(cam_name + '_%03d' %(i+1) + '_pred.png')
    gif_name = cam_name + '_gif.gif'
    create_gif(image_list, gif_name)

def main():
    image_list_full = os.listdir(img_directory)
    gather_image_list('cam_00_10_30', image_list_full)
    gather_image_list('cam_01_10_30', image_list_full)
    gather_image_list('cam_02_10_30', image_list_full)
    gather_image_list('cam_03_10_30', image_list_full)

if __name__ == "__main__":
    main()
