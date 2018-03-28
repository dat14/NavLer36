from asset_import import *

batch_size = 32
max_iterations = 200

# Set this path to your dataset directory
path = "C:/Users/trand/Desktop/PseudoCorridor/Merged/"

directory = "C:/Users/trand/Desktop/PseudoCorridor/Merged/"
dataset = "all_groups.csv"


class datasource(object):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses


class train_datasource(object):
    def __init__(self, train_images, train_poses):
        self.train_images = train_images
        self.train_poses = train_poses


class validation_datasource(object):
    def __init__(self, validation_images, validation_poses):
        self.validation_images = validation_images
        self.validation_poses = validation_poses


class test_datasource(object):
    def __init__(self, test_images, test_poses):
        self.test_images = test_images
        self.test_poses = test_poses


class datasource_split(object):
    def __init__(self, train_datasource, validation_datasource, test_datasource):
        self.train = train_datasource
        self.validation = validation_datasource
        self.test = test_datasource


def centeredCrop(img, output_side_length1, output_side_length2):
    height, width, depth = img.shape
    new_height = output_side_length1
    new_width = output_side_length2
    height_offset = (new_height - output_side_length1) / 2
    width_offset = (new_width - output_side_length2) / 2
    cropped_img = img[height_offset:height_offset + output_side_length1,
                  width_offset:width_offset + output_side_length2]
    return cropped_img


def preprocess(image):
    # Resize and crop and compute mean!
    with tf.device('/cpu:0'):
            image_out = cv2.imread(image)
        return image_out


def get_data():
    poses = []
    images_list = []
    with open(directory + dataset) as f:
        for line in tqdm(f):
            fname, p0, p1, p2, p3, p4 = line.split(',')
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            #			p5 = 1
            #			p5 = float(p5)
            #			p6 = 1
            #			p6 = float(p6)
            poses.append([p0, p1, p2, p3, p4
                          # ,p5,p6
                          ])
            images_list.append(directory + fname)
    return datasource(images_list, poses)


def train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=None):
    np.random.seed(seed)
    perm = list(range(len(df.images)))
    random.shuffle(perm)

    print(type(df.images))

    m = len(df.images)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end

    print("m: %d" % m)
    print(type(m))

    for i in perm:
        images = df.images[i]
        poses = df.poses[i]

    train_images = df.images[:train_end]
    print("train_images lenght: %d" % len(train_images))

    validation_images = df.images[train_end:validate_end]
    print("validation_images lenght: %d" % len(validation_images))

    test_images = df.images[validate_end:]
    print("test_images lenght: %d" % len(test_images))

    train_poses = df.poses[:train_end]
    print("train_poses lenght: %d" % len(train_poses))

    validation_poses = df.poses[train_end:validate_end]
    print("validation_poses lenght: %d" % len(validation_poses))

    test_poses = df.poses[validate_end:]
    print("test_poses lenght: %d" % len(test_poses))

    return train_datasource(train_images, train_poses), validation_datasource(validation_images,
                                                                              validation_poses), test_datasource(
        test_images, test_poses)


def gen_data(source):
    while True:
        indices = list(range(len(source.train_images)))
        random.shuffle(indices)
        for i in indices:
            image = source.train_images[i]
            pose_x = source.train_poses[i][0:2]
            pose_q = source.train_poses[i][2:5]
            yield image, pose_x, pose_q


def gen_data_val(source):
    indices = list(range(len(source.validation_images)))
    for i in indices:
        image = source.validation_images[i]
        pose = source.validation_poses[i]
        yield image, pose


def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            preprocess(image)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)


def gen_data_batch_val(source):
    data_gen = gen_data_val(source)
    while True:
        image_batch = []
        pose_val_batch = []
        for _ in range(batch_size):
            image, pose_val = next(data_gen)
            image_batch.append(image)
            pose_val_batch.append(pose_val)
        yield np.array(image_batch), np.array(pose_val_batch)
