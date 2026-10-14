import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from timm.loss import LabelSmoothingCrossEntropy
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
# Add this code to your main script where you need to load the data


def setup_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
def load_data_houston(data_path, label_path=None):
    """Load and normalize target-domain Houston data and its label map.

    Args:
        data_path: Path to the hyperspectral data file.
        label_path: Optional separate label file; defaults to data_path.

    Returns:
        input_normalize: Band-wise standardized image cube.
        label: Ground-truth label map.
    """
    print(f"Loading target domain data from {data_path}")
    
    # Load the hyperspectral data file.
    data = loadmat(data_path)
    
    # Identify the label array using the available Houston18 field names.
    if label_path is None:
        # Read labels from the data file when no separate label file is supplied.
        if 'houston18hsi' in data:
            label = data['houston18hsi']
            print("Found 'houston18hsi' label in data file")
        elif 'TR' in data and 'TE' in data:
            TR = data['TR']
            TE = data['TE']
            label = TR + TE
        elif 'label' in data:
            label = data['label']
        else:
            raise ValueError("Label not found in data file, please provide label_path")
    else:
        # Read labels from the separate label file.
        print(f"Loading target domain labels from {label_path}")
        label_data = loadmat(label_path)
        if 'houston18hsi' in label_data:
            label = label_data['houston18hsi']
        elif 'label' in label_data:
            label = label_data['label']
        else:
            # Use the first variable after the three MATLAB metadata entries.
            label_key = list(label_data.keys())[3]
            label = label_data[label_key]
    
    # Identify the hyperspectral input cube.
    if 'ori_data' in data:
        input_data = data['ori_data']
        print("Found 'ori_data' in data file")
    elif 'input' in data:
        input_data = data['input']
    else:
        # Use the first variable after the three MATLAB metadata entries.
        input_key = list(data.keys())[3]
        input_data = data[input_key]
    
    # Standardize each spectral band independently.
    input_normalize = np.zeros(input_data.shape)
    for i in range(input_data.shape[2]):
        input_normalize[:,:,i] = (input_data[:,:,i] - np.mean(input_data[:,:,i])) / np.std(input_data[:,:,i])
    
    print(f"Target data shape: {input_normalize.shape}")
    print(f"Target label shape: {label.shape}")
    print(f"Target data classes: {np.unique(label[label > 0])}")
    
    return input_normalize, label

def get_all_houston18_data(input_data, label, patch_size):
    """Extract labeled target-domain test patches and their metadata.

    Args:
        input_data: Normalized hyperspectral image cube.
        label: Ground-truth label map.
        patch_size: Spatial neighborhood width and height.

    Returns:
        testID: Sequential test-sample identifiers.
        testX: Test patches.
        testY: Zero-based test labels.
        G: Labeled spatial grid.
        RandPerm: Random permutation of test samples.
        Row: Image height.
        Column: Image width.
    """
    height, width, band = input_data.shape
    padding = patch_size // 2
    
    # Mirror-pad the image before patch extraction.
    mirror_image = mirror_hsi(height, width, band, input_data, patch=patch_size)
    
    # Locate all labeled pixels.
    test_positions = np.argwhere(label > 0)
    num_test = test_positions.shape[0]
    
    # Allocate one spatial patch for each labeled test pixel.
    testX = np.zeros((num_test, patch_size, patch_size, band), dtype=float)
    testY = np.zeros(num_test, dtype=int)
    testID = np.zeros(num_test, dtype=int)
    
    # Extract patches and convert labels to zero-based indices.
    for i in range(num_test):
        x, y = test_positions[i]
        testX[i] = mirror_image[x:x+patch_size, y:y+patch_size, :]
        testY[i] = label[x, y] - 1  # MATLAB labels are one-based.
        testID[i] = i
    
    # Build a spatial label grid for optional visualization.
    G = np.zeros((height, width))
    for i in range(num_test):
        x, y = test_positions[i]
        G[x, y] = testY[i] + 1
    
    # Generate a permutation for optional sample shuffling.
    RandPerm = np.random.permutation(num_test)
    
    # Record image dimensions.
    Row, Column = height, width
    
    return testID, testX, testY, G, RandPerm, Row, Column

def load_HSI(args):
    """Load an image cube, class masks, normalized data, and color tables."""
    if args.dataset == 'Indian':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/IndianPine.mat')
    elif args.dataset == 'paviaU':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/paviaU_0709_30.mat')
    elif args.dataset == 'pavia':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/pavia_split.mat')
    elif args.dataset == 'Houston13':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/Houston13_split_0609180.mat')
    elif args.dataset == 'Houston':
        # data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/Houston13_split_060830.mat')
        data = loadmat('/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型/houston_30.mat')
    elif args.dataset == 'Houston18':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/Houston18_split.mat')
    elif args.dataset == 'Dioni':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/Dioni_0709_30.mat')
    elif args.dataset == 'Loukia':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/Loukia_split.mat')
    elif args.dataset == 'HZ':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/HZ_split.mat')
    elif args.dataset == 'SH':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/shanghai_0710_30.mat')
    elif args.dataset == 'IP1':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/Indiana_0712_30.mat')
    elif args.dataset == 'IP2':
        data = loadmat('/mnt/HDD/data/wxyxier/S2MABA/data/Indiana_split.mat')
    elif args.dataset == 'WHU_Hi_LongKou':
        data = loadmat('./data/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        TR = loadmat('./data/WHU-Hi-LongKou/TRTE/Train25.mat')['LKtrain25']
        TE_class = loadmat('./data/WHU-Hi-LongKou/TRTE/Test25.mat')['LKtest25']
    else:
        raise ValueError("Unknown dataset")

    if args.dataset != 'WHU_Hi_LongKou':
        TR = data['TR']               # Training class-index mask.
        TE_class = data['TE']         # Test class-index mask.
        input_img = data['input']     # Raw hyperspectral image cube.
    else:
        input_img = data

    # Convert the test class mask to a binary sample mask.
    TE_mask = (TE_class > 0).astype(np.uint8)

    # Combine the train and test masks into a one-based label map.
    label = TR + TE_class
    num_classes = int(np.max(label))

    # Standardize each spectral band of the input cube.
    input_normalize = np.zeros_like(input_img, dtype=np.float32)
    for b in range(input_img.shape[2]):
        band_data = input_img[:, :, b]
        input_normalize[:, :, b] = (band_data - band_data.mean()) / band_data.std()

    # Define color tables for optional visualization.
    color_matrix = [
        [0,0,0], [83,171,72], [137,186,67], [66,132,91], [60,131,69],
        [144,82,54], [105,188,200], [255,255,255], [199,176,201], [218,51,44],
        [119,35,36], [55,101,166], [224,219,84], [217,142,52], [84,48,126],
        [227,119,91], [157,87,150]
    ]
    color_matrix_pred = color_matrix[1:]
    color_matrix = [[c/256. for c in color] for color in color_matrix]
    color_matrix_pred = [[c/256. for c in color] for color in color_matrix_pred]

    return input_normalize, label, num_classes, TR, TE_mask, TE_class, color_matrix, color_matrix_pred

    
def choose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    
    # Handle training data
    total_pos_train = np.empty((0, 2), dtype=int)  # Initialize with empty array of correct shape
    for i in range(num_classes):
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
        if each_class.shape[0] > 0:  # Only concatenate if there are points
            total_pos_train = np.r_[total_pos_train, each_class]
    
    # Handle testing data
    total_pos_test = np.empty((0, 2), dtype=int)  # Initialize with empty array of correct shape
    for i in range(num_classes):
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class
        if each_class.shape[0] > 0:  # Only concatenate if there are points
            total_pos_test = np.r_[total_pos_test, each_class]
    
    # Handle true data
    total_pos_true = np.empty((0, 2), dtype=int)  # Initialize with empty array of correct shape
    for i in range(num_classes+1):
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class
        if each_class.shape[0] > 0:  # Only concatenate if there are points
            total_pos_true = np.r_[total_pos_true, each_class]
    
    total_pos_train = total_pos_train.astype(int)
    total_pos_test = total_pos_test.astype(int)
    total_pos_true = total_pos_true.astype(int)
    
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]
    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi

def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image


def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    patch_size = 25  # Retained for compatibility with the original routine.
    pad = patch_size // 2
    
    for i in range(train_point.shape[0]):
        # mirror_image = np.pad(mirror_image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
        # print(f"Index {i}: patch_data.shape = {x_train[i,:,:,:].shape}")
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    # for k in range(true_point.shape[0]):
    #     x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    # print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    return x_train, x_test#, x_true#

def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
import numpy as np
def choose_train_and_test_point_t(TR, TE, label, num_classes):
    """Identify labeled Houston18 test pixels without target-domain training.

    Args:
        TR: Target-domain training mask, unused in this setting.
        TE: Binary target-domain test mask.
        label: One-based ground-truth label map.
        num_classes: Number of target classes.

    Returns:
        total_pos_train: Empty training positions for every class.
        total_pos_test: Test positions grouped by class.
        total_pos_true: All labeled positions grouped by class.
        number_train: Zero training counts for every class.
        number_test: Test counts by class.
        number_true: Labeled counts by class.
    """
    total_pos_true = {}
    total_pos_train = {}  # No target-domain samples are used for training.
    total_pos_test = {}
    number_train = np.zeros(num_classes)  # Target-domain training counts are zero.
    number_test = np.zeros(num_classes)
    number_true = np.zeros(num_classes)
    
    # Locate all labeled samples for each class.
    for i in range(num_classes):
        idx = np.where(label == i + 1)
        total_pos_true[i] = list(zip(idx[0], idx[1]))
        number_true[i] = len(total_pos_true[i])
        
        # Assign labeled pixels in TE to the test set.
        class_pos = np.where((TE == 1) & (label == i + 1))
        test_pos = list(zip(class_pos[0], class_pos[1]))
        
        # Record the test positions when this class is represented.
        if len(test_pos) > 0:
            total_pos_test[i] = test_pos
            number_test[i] = len(test_pos)
            
        # Exclude target-domain samples from training.
        total_pos_train[i] = []
        number_train[i] = 0
    
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def train_and_test_data_t(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=5):
    """Extract Houston18 test patches for the cross-domain setting.

    Args:
        mirror_image: Mirror-padded hyperspectral image cube.
        band: Number of spectral bands.
        total_pos_train: Unused target-domain training positions.
        total_pos_test: Test positions grouped by class.
        total_pos_true: All labeled positions grouped by class.
        patch: Spatial patch size.

    Returns:
        x_train: Empty target-domain training array.
        x_test: Target-domain test patches.
    """
    # Compute the spatial padding width.
    pad = patch // 2
    
    # Leave the target-domain training set empty.
    x_train = np.zeros((0, patch, patch, band))
    
    # Allocate the target-domain test set.
    test_samples_count = sum(len(pos) for pos in total_pos_test.values())
    x_test = np.zeros((test_samples_count, patch, patch, band))
    
    # Populate test patches in class order.
    test_idx = 0
    for class_idx, positions in total_pos_test.items():
        for pos in positions:
            x, y = pos
            # Extract the patch centered on this labeled pixel.
            x_test[test_idx] = mirror_image[x:x+patch, y:y+patch, :]
            test_idx += 1
    
    return x_train, x_test


def train_and_test_label_t(number_train, number_test, number_true, num_classes):
    """Construct zero-based Houston18 labels for cross-domain evaluation.

    Args:
        number_train: Zero target-domain training counts by class.
        number_test: Test counts by class.
        number_true: Labeled counts by class.
        num_classes: Number of target classes.

    Returns:
        y_train: Empty target-domain training labels.
        y_test: Zero-based target-domain test labels.
        y_true: Zero-based labels for all labeled pixels.
    """
    # Leave the target-domain training labels empty.
    y_train = np.zeros(0)
    
    # Use zero-based test labels for compatibility with cross-entropy loss.
    # The input masks encode class indices starting from one.
    y_test = np.zeros(int(sum(number_test)))
    test_idx = 0
    for i in range(num_classes):
        if number_test[i] > 0:
            y_test[test_idx:test_idx+int(number_test[i])] = i  # Zero-based class index.
            test_idx += int(number_test[i])
    
    # Construct labels for every annotated pixel.
    y_true = np.zeros(int(sum(number_true)))
    true_idx = 0
    for i in range(num_classes):
        if number_true[i] > 0:
            y_true[true_idx:true_idx+int(number_true[i])] = i  # Zero-based class index.
            true_idx += int(number_true[i])
    
    return y_train, y_test, y_true

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()


def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   
        optimizer.zero_grad()
        batch_pred,_ = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()       
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()  
        batch_pred,feature = model(batch_data)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre,feature

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()  
        batch_pred = model(batch_data)
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

def output_metric(tar, pre, num_classes=None):
    """Return aggregate and class-specific accuracies without printing the matrix."""
    if num_classes is not None:
        if np.any(np.asarray(pre) < 0) or np.any(np.asarray(pre) >= num_classes):
            raise ValueError("Predicted class indices exceed the target class range")
        labels = range(num_classes)
    else:
        labels = None
    matrix = confusion_matrix(tar, pre, labels=labels)
    if np.sum(matrix) == 0:
        raise ValueError("The evaluation set contains no labeled samples")
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.full([shape[0]], np.nan, dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        class_total = np.sum(matrix[i, :])
        if class_total > 0:
            AA[i] = matrix[i, i] / class_total
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.nanmean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
