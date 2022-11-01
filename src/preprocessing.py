import os
import argparse
def move(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    if not os.path.exists(os.path.join(dst, 'blur')):
        os.mkdir(os.path.join(dst, 'blur'))
    if not os.path.exists(os.path.join(dst, 'sharp')):
        os.mkdir(os.path.join(dst, 'sharp'))

    folders = os.listdir(src)
    cnt = 0
    for f in folders:
        image_names = os.listdir(os.path.join(src, f, 'blur'))

        for i in image_names:
            os.rename(os.path.join(src, f, 'blur', i), os.path.join(dst, 'blur', f + '_' + i))
            os.rename(os.path.join(src, f, 'sharp', i), os.path.join(dst, 'sharp', f + '_' + i))
            cnt += 1
    print('%d images are moved' % cnt)

    
def preprocess_dataset(root_src='dataset/GOPRO_Large',  root_dst='dataset/GOPRO'):
    os.makedirs(root_dst)
    move(os.path.join(root_src, 'train'), os.path.join(root_dst, 'train'))
    move(os.path.join(root_src, 'test'), os.path.join(root_dst, 'test'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_src', default='dataset/GOPRO_Large', type=str)
    parser.add_argument('--root_dst', default='dataset/GOPRO', type=str)
    args = parser.parse_args()
    
    preprocess_dataset(args.root_src, args.root_dst)