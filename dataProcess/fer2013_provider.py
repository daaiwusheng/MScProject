from utility.tools import *
import os


class FER2013Provider(object):
    def __init__(self):
        self.dict_emotion_2_label = {'neutral': 0,
                                     'angry': 1,
                                     'contempt': 2,
                                     'disgust': 3,
                                     'fear': 4,
                                     'happy': 5,
                                     'sad': 6,
                                     'surprise': 7}

        # on Mac
        # self.root_dir = '/home/steven/桌面/AICode/project_dataset/FER2013/test'
        # self.save_dir = "/home/steven/桌面/AICode/project_dataset/FER2013/savehandeldata/"

        # on Linux
        self.root_dir = '/home/steven/桌面/AICode/project_dataset/FER2013/test'
        self.save_dir = "/home/steven/桌面/AICode/project_dataset/FER2013/savehandeldata/"

        self.save_validate_dir = self.save_dir + 'fer2013_validate.csv'

        self.array_images = []  # image dir list
        self.array_labels = []  # label list
        data_exist_bool = os.path.exists(self.save_validate_dir)

        if not data_exist_bool:
            self.dict_emotion_images = {}  # key is emotion, v is image name
            self.get_dict_emotion_images()  # key is emotion, v is imagenames list
            self.dict_image_label = {}  # key is image dir, v is label
            self.get_dict_image_label()
            self.store_images_labels()
            save_dict_as_csv(self.save_validate_dir, self.dict_image_label)

        self.__get_data_for_outside(self.array_images, self.array_labels, self.save_validate_dir)

    def get_data(self, index):
        image = self.array_images[index]
        label = self.array_labels[index]
        return image, label

    def __len__(self):
        return len(self.array_images)

    def __get_data_for_outside(self, array_img, array_label, save_path):
        _dict_image_dir_emotion = load_dict_from_csv(save_path)
        for image_dir, emotion in _dict_image_dir_emotion.items():
            img = cv2.imread(image_dir, 0)
            label_emotion = int(emotion)
            array_img.append(img)
            array_label.append(label_emotion)

    def store_images_labels(self):
        for k, v in self.dict_image_label.items():
            self.array_images.append(k)
            self.array_labels.append(v)

    def get_dict_image_label(self):
        for k, images in self.dict_emotion_images.items():
            image_label = self.dict_emotion_2_label[k]
            for img in images:
                img_dir = os.path.join(self.root_dir, k, img)
                self.dict_image_label[img_dir] = image_label

    def get_dict_emotion_images(self):
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            for emotion in dirs:
                # print(os.path.join(root, emotion))
                for _, _, image_names in os.walk(os.path.join(root, emotion), topdown=True):
                    self.dict_emotion_images[emotion] = image_names
