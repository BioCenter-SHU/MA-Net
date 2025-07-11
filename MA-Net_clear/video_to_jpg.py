from __future__ import print_function, division
import os
import sys
import subprocess

# def class_process(dir_path, dst_dir_path, class_name):
# #   class_path = os.path.join(dir_path, class_name)
# #   if not os.path.isdir(class_path):
# #     return

#   dst_class_path = os.path.join(dst_dir_path, class_name)
#   if not os.path.exists(dst_class_path):
#     os.mkdir(dst_class_path)

#   for file_name in os.listdir(class_path):
#     # if '.avi' not in file_name:
#     #   continue
#     name, ext = os.path.splitext(file_name)
#     dst_directory_path = os.path.join(dst_class_path, name)

#     video_file_path = os.path.join(class_path, file_name)
#     try:
#       if os.path.exists(dst_directory_path):
#         if not os.path.exists(os.path.join(dst_directory_path, 'image_000001.jpg')):
#           subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
#           print('remove {}'.format(dst_directory_path))
#           os.mkdir(dst_directory_path)
#         else:
#           continue
#       else:
#         os.mkdir(dst_directory_path)
#     except:
#       print(dst_directory_path)
#       continue
#     cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%03d.jpg\"'.format(video_file_path, dst_directory_path)
#     #cmd = 'ffmpeg -i \"{}\" -vf fps=fps=1/60 -qscale:v 2 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
#     print(cmd)
#     subprocess.call(cmd, shell=True)
#     print('\n')

import torch.nn.functional as F
F.interpolate()

def class_process(dir_path, dst_dir_pathe):
  for file_name in os.listdir(dir_path):
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_dir_pathe, name)

    video_file_path = os.path.join(dir_path, file_name)
    try:
      if os.path.exists(dst_directory_path):
        if not os.path.exists(os.path.join(dst_directory_path, 'image_000001.jpg')):
          subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
          print('remove {}'.format(dst_directory_path))
          os.mkdir(dst_directory_path)
        else:
          continue
      else:
        os.mkdir(dst_directory_path)
    except:
      print(dst_directory_path)
      continue
    cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%03d.jpg\"'.format(video_file_path, dst_directory_path)
    #cmd = 'ffmpeg -i \"{}\" -vf fps=fps=1/60 -qscale:v 2 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')


if __name__=="__main__":
    dir_path = "/YOUR_PATH/dataset/patient_video/"
    dst_dir_path = '/YOUR_PATH/ACTION-Net-main/data/frame/'
    class_process(dir_path, dst_dir_path)
  # for video_name in os.listdir(dir_path):
    # class_process(dir_path, dst_dir_path, video_name)


