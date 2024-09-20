# -*- coding:utf-8 -*-
import utils2

img_dir1 = ''   # path to cover folder
img_dir2 = ''   # path to stego folder

print(utils2.psnr_between_dirs(img_dir1, img_dir2))
print(utils2.ssim_between_dirs(img_dir1, img_dir2, True))
print(utils2.vgg_between_dirs(img_dir1, img_dir2, True))
print(utils2.inception_score_in_folder(img_dir2))

