import cv2
import torch


def outclude_hidden_files(files):
    return [f for f in files if not f[0] == '.']


def outclude_hidden_dirs(dirs):
    return [d for d in dirs if not d[0] == '.']


def show_image(image, window_name='test'):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()