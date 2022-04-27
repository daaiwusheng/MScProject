import os
import random

from utility.tools import *
import numpy as np

# classes = ['Neutral - NE', 'Anger - AN', 'Contempt - CO', 'Disgust - DI', 'Fear - FR', 'Happiness - HA', 'Sadness - SA', 'Surprise - SU']
Neutral = 0
Anger = 1
Contempt = 2
Disgust = 3
Fear = 4
Happiness = 5
Sadness = 6
Surprise = 7

emotion_labs = [Neutral, Anger, Contempt, Disgust, Fear, Happiness, Sadness, Surprise]


split_factor = 0.8

number_frames = 2


class CKPDataProvider(object):
    def __init__(self, reconstruct=False, is_train=True):
        # on Linux
        self.is_train = is_train
        self.root_dir = '/home/steven/code/MScProject_dataset/CK+/Emotion'
        self.r_dir_images = '/home/steven/code/MScProject_dataset/CK+/cohn-kanade-images/'
        self.r = '/home/steven/code/MScProject_dataset/'
        self.save_dir = "/home/steven/code/savehandeldata/"
        self.save_train_dir = self.save_dir + 'ck_train.csv'
        self.save_validate_dir = self.save_dir + 'ck_validate.csv'

        data_exist_bool = os.path.exists(self.save_train_dir) & os.path.exists(self.save_validate_dir)

        if reconstruct:
            self.__constrrct_data()
        elif not data_exist_bool:
            self.__constrrct_data()
        else:
            pass

        self.array_train_images = []  # store train image array data
        self.array_train_labels = []
        self.__get_train_data_for_outside(self.array_train_images, self.array_train_labels, self.save_train_dir)
        self.len_train_data = len(self.array_train_labels)

        self.array_validate_images = []  # store validate image array data
        self.array_validate_labels = []
        self.__get_train_data_for_outside(self.array_validate_images, self.array_validate_labels,
                                          self.save_validate_dir)
        self.len_validate_data = len(self.array_validate_labels)

    def get_data(self, index):
        if self.is_train:
            index = index % self.len_train_data
            return self.array_train_images[index], self.array_train_labels[index]
        else:
            index = index % self.len_validate_data
            return self.array_validate_images[index], self.array_validate_labels[index]

    def __len__(self):
        if self.is_train:
            return len(self.array_train_images)
        else:
            return len(self.array_validate_images)

    def __get_train_data_for_outside(self, array_img, array_label, save_path):
        _dict_image_dir_emotion = load_dict_from_csv(save_path)
        for image_dir, emotion in _dict_image_dir_emotion.items():
            image_full_path = os.path.join(self.r, image_dir)
            img = cv2.imread(image_full_path, 0)
            label_emotion = int(emotion)
            array_img.append(img)
            array_label.append(label_emotion)

    def __constrrct_data(self):
        self.dirs_emotion_actors = self.get_dirs_emotion_actors()
        # actor means the code like S506,
        # every emotion dir is an actor's emotion dir, in which it's maybe more than one emotions
        self.actors_list = list()
        self.get_all_actors()
        # 找到每个演员所拥有的有效表情，也就是标签
        # 也找到每个表情所对应的演员
        # 找到每个图片所对应的标签
        self.dict_key_seq_labels = {}  # 字典,存储key值,value是标签值
        self.key_sequence_with_labels = []  # 存储 key值 (如S506/002),按这个key就可以索引出标签,图片,landmarks,而不会乱掉
        self.get_necessary_data()
        self.dict_actor_labels = {}  # key is actor s160, v is array including labels
        self.get_dict_actor_labels()
        self.dict_emotion_actors = {}  # key is emotion, v is array including actors
        self.get_dict_emotion_actors()
        self.dict_train_emotion_actors = {}  # train set
        self.dict_validate_emotion_actors = {}  # validation set
        self.whole_train_actors_list = []  # store all train actors
        self.handle_train_validate_sets()
        # self.test_train_actors()  # test if the train data is split correctly
        self.whole_validate_actors_list = []  # store all validate actors
        self.get_validate_actors_list()
        # 根据actor把图片key值拿到， 然后取出最后2 frames
        # 然后图片名作为key，label作为v
        self.dict_train_key_emotion = {}  # key is S011/001, v is emotion label
        self.get_dict_train_key_emotion()
        self.dict_validate_key_emotion = {}  # key is S011/001, v is emotion label
        self.get_dict_validate_key_emotion()
        self.dict_train_image_filename_emotion = {}  # key is image file name, v is emotion label
        self.neutral_train_image_files = []
        self.get_dict_train_image_filename_emotion()
        self.dict_validate_image_filename_emotion = {}  # key is image file name, v is emotion label
        self.neutral_validate_image_files = []
        self.get_dict_validate_image_filename_emotion()
        self.get_neutral_img_for_train_val()

        self.dict_train_image_dir_emotion = {}  # key is like "CK+/cohn-kanade-images/S005/001/S005_001_00000001.png"
        self.dict_validate_image_dir_emotion = {}
        self.get_image_dir_dict(self.dict_train_image_filename_emotion, self.dict_train_image_dir_emotion)
        self.get_image_dir_dict(self.dict_validate_image_filename_emotion, self.dict_validate_image_dir_emotion)

        save_dict_as_csv(self.save_train_dir, self.dict_train_image_dir_emotion)
        save_dict_as_csv(self.save_validate_dir, self.dict_validate_image_dir_emotion)

    def get_image_dir_dict(self, dict_image_file_emotion, dict_result):
        relative_dir_r = "CK+/cohn-kanade-images/"
        for image_file, emotion in dict_image_file_emotion.items():
            image_file_splits = image_file.split('_')
            dir_image = image_file_splits[0] + '/' + image_file_splits[1]
            image_dir = os.path.join(relative_dir_r + dir_image, image_file)
            dict_result[image_dir] = emotion

    def get_neutral_img_for_train_val(self):
        #  randomly get neutral images for train and validate sets
        num_classes_out_neutral = 7
        random.shuffle(self.neutral_train_image_files)
        num_train_neutral = int(len(self.dict_train_image_filename_emotion) / num_classes_out_neutral)
        images_need_train_neutral = self.neutral_train_image_files[:num_train_neutral]

        for image_file_name in images_need_train_neutral:
            self.dict_train_image_filename_emotion[image_file_name] = 0

        random.shuffle(self.neutral_validate_image_files)
        num_validate_neutral = int(len(self.dict_validate_image_filename_emotion) / num_classes_out_neutral)
        images_need_validate_neutral = self.neutral_validate_image_files[:num_validate_neutral]

        for image_file_name in images_need_validate_neutral:
            self.dict_validate_image_filename_emotion[image_file_name] = 0

        # print("line",get_line_num(),len(images_need_train_neutral)," ",len(images_need_validate_neutral))

    def get_dict_validate_image_filename_emotion(self):
        for ikey, emotion in self.dict_validate_key_emotion.items():
            for r, _, image_files in os.walk(os.path.join(self.r_dir_images, ikey)):
                image_files = outclude_hidden_files(image_files)
                image_files.sort()
                image_files_need = image_files[-number_frames:]
                self.neutral_validate_image_files.append(image_files[0])
                for image_file in image_files_need:
                    self.dict_validate_image_filename_emotion[image_file] = emotion

    def get_dict_train_image_filename_emotion(self):
        for ikey, emotion in self.dict_train_key_emotion.items():
            for r, _, image_files in os.walk(os.path.join(self.r_dir_images, ikey)):
                image_files = outclude_hidden_files(image_files)
                image_files.sort()
                image_files_need = image_files[-number_frames:]
                self.neutral_train_image_files.append(image_files[0])
                for image_file in image_files_need:
                    self.dict_train_image_filename_emotion[image_file] = emotion

    def get_dict_validate_key_emotion(self):
        for actor in self.whole_validate_actors_list:
            for k, emotion in self.dict_key_seq_labels.items():
                k_actor = k.split('/')[0]
                if k_actor == actor:
                    self.dict_validate_key_emotion[k] = emotion

    def get_dict_train_key_emotion(self):
        for actor in self.whole_train_actors_list:
            for k, emotion in self.dict_key_seq_labels.items():
                k_actor = k.split('/')[0]
                if k_actor == actor:
                    self.dict_train_key_emotion[k] = emotion

    def get_validate_actors_list(self):
        self.whole_validate_actors_list = list(set(self.actors_list).difference(set(self.whole_train_actors_list)))
        # print(len(self.actors_list))
        # print(len(self.whole_validate_actors_list))
        # print(len(self.whole_train_actors_list))

    def handle_train_validate_sets(self):
        # first, sort emotions by the relative number of actors
        array_order = np.array([2, 4, 6, 1, 3, 5, 7])  # [18,25,28,45,59,69,83]
        factor_test = 1 - split_factor
        for emotion in array_order:
            # if emotion in self.dict_train_emotion_actors:
            # actors_train_list = []
            actors = self.dict_emotion_actors[emotion]
            actors_not_in_train = list(set(actors).difference(set(self.whole_train_actors_list)))
            # actors_in_train = list(set(whole_train_actors_list).difference(set(actors)))
            factor_not_in_train = round(len(actors_not_in_train) / len(actors), 2)
            if factor_not_in_train > factor_test:
                factor_move_to_train = factor_not_in_train - factor_test
                number_move_to_train = int(factor_move_to_train * len(actors))
                # move
                self.whole_train_actors_list = self.whole_train_actors_list + actors_not_in_train[:number_move_to_train]
                # self.dict_train_emotion_actors[emotion] = actors_train_list

    def test_train_actors(self):
        # produce dict_train_emotion_actors
        for actor in self.whole_train_actors_list:
            emotions_list = self.dict_actor_labels[actor]
            for emotion in emotions_list:
                if emotion in self.dict_train_emotion_actors:
                    actors_list = self.dict_train_emotion_actors[emotion]
                    actors_list.append(actor)
                else:
                    actors_list = list([actor])
                    self.dict_train_emotion_actors[emotion] = actors_list
        for emotion, actors in self.dict_emotion_actors.items():
            actors_train_list = self.dict_train_emotion_actors[emotion]
            factor = round(len(actors_train_list) / len(actors), 2)
            print(emotion, " ", factor)
            print(actors)
            print(actors_train_list)

    def get_dict_emotion_actors(self):
        for actor, emotions_set in self.dict_actor_labels.items():
            for emotion in emotions_set:
                if emotion in self.dict_emotion_actors:
                    actors_set = self.dict_emotion_actors[emotion]
                    actors_set.add(actor)
                else:
                    actors_set = set()
                    actors_set.add(actor)
                    self.dict_emotion_actors[emotion] = actors_set
        for emotion, actors in self.dict_emotion_actors.items():
            actors_list = list(actors)
            self.dict_emotion_actors[emotion] = actors_list

    def get_dict_actor_labels(self):
        for k, v in self.dict_key_seq_labels.items():
            actor = k.split('/')[0]
            if actor not in self.dict_actor_labels:
                emotions_set = set()
                emotions_set.add(v)
                self.dict_actor_labels[actor] = emotions_set
            else:
                emotions_set = self.dict_actor_labels[actor]
                emotions_set.add(v)
        for actor, labels in self.dict_actor_labels.items():
            labels_list = list(labels)
            self.dict_actor_labels[actor] = labels_list

    def get_necessary_data(self):
        emotion_files = []  # 存储表情标签的txt 文件的路径
        labels = []
        for emotion_actor in self.dirs_emotion_actors:
            # print(emotion_actor)
            for root, dir_frames, _ in os.walk(emotion_actor):
                dir_frames = outclude_hidden_dirs(dir_frames)
                dir_frames.sort()
                # print(os.path.join(root, dir_frames[0]))
                for dir_frame in dir_frames:
                    for r, dir_emotion, emotion_file in os.walk(os.path.join(root, dir_frame)):
                        emotion_file = outclude_hidden_files(emotion_file)
                        if emotion_file:
                            r_split = r.split('/')
                            key_ = r_split[-2] + '/' + r_split[-1]
                            self.key_sequence_with_labels.append(key_)
                            emotion_files.append(os.path.join(r, emotion_file[0]))
                            f = open(emotion_files[-1], 'r+')
                            line = f.readline()  # only one row
                            # print(int(line.split('.')[0]))
                            label = int(line.split('.')[0])
                            labels.append(label)
                            self.dict_key_seq_labels[key_] = label
                break

    def get_dirs_emotion_actors(self):
        dirs_emotion_actors = []
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            dirs = outclude_hidden_dirs(dirs)
            dirs.sort()
            for name in dirs:
                # print(os.path.join(root, name))
                dirs_emotion_actors.append(os.path.join(root, name))
            break

        return dirs_emotion_actors

    def get_all_actors(self):
        for dir_actor in self.dirs_emotion_actors:
            actor = dir_actor.split('/')[-1]
            self.actors_list.append(actor)
