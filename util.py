import numpy as np
import tensorflow as tf
import cv2

def read_labels(path_to_labels, one_hot = False):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.int8)
    
    if not one_hot:
        return labels
    
    one_hot_label = np.zeros((labels.shape[0], 10))
    
    for i, index in enumerate(labels):
        one_hot_label[i][index - 1] = 1
    
    return one_hot_label

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def save_cam(cam, image):
    
    image = np.uint8(image[:, :, ::-1] * 255.0) # RGB -> BGR
    cam = cv2.resize(cam, (96, 96)) # enlarge heatmap
    heatmap = cam / np.max(cam) # normalize
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # balck-and-white to color
    cam = np.float32(cam) + np.float32(image) # everlay heatmap onto the image
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)
    
    return cam

    
def data_aug(img, aug_width, aug_height):
    
    return np.array([cv2.resize(ele, (aug_width, aug_height), interpolation = cv2.INTER_LANCZOS4) for ele in img])

def extract_patch(img, args):
    ih, iw, c = img.shape
    ix = np.random.randint(iw - args.width + 1, size = 1)[0]
    iy = np.random.randint(ih - args.height + 1, size = 1)[0]
    img = img[iy: iy + args.height, ix : ix + args.width]
    
    flip_lr = np.random.randint(6, size = 1)[0]
    if flip_lr % 2 == 0:
        return np.fliplr(img)
    
    return img
        
def train_batch_gen(img, args, r_index, k):
    
    batch_img = img[r_index[args.batch_size * k : args.batch_size * (k + 1)]]
    
    return np.array([extract_patch(ele, args) for ele in batch_img])
    
    

def split_data(img, label, ratio = 0.8):
    
    dic = {}
    
    for i, ele in enumerate(label):
        if ele in dic.keys():
            dic[ele] = np.append(dic[ele], i)
        else:
            dic[ele] = np.array((i))
    
    te_img = []
    te_label = []        
    val_img = []
    val_label = []
    
    for key in dic.keys():
        size = len(dic[key])
        te_size = int(size * ratio)
        te_img.extend(img[dic[key][:te_size]])
        
        zero_mat = np.zeros((te_size, 10))
        zero_mat[:, key - 1] = 1
        
        te_label.extend(zero_mat)
        
        val_img.extend(img[dic[key][te_size:]])
        
        zero_mat = np.zeros((size - te_size, 10))
        zero_mat[:, key - 1] = 1
        
        val_label.append(zero_mat)
        
    
    te_img = np.array(te_img)
    val_img = np.array(val_img)
    te_label = np.reshape(np.array(te_label), (-1, 10))
    val_label = np.reshape(np.array(val_label), (-1, 10))

    return te_img, te_label, val_img, val_label

