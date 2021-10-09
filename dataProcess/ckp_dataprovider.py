import os
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


class CKPDataProvider(object):
    def __init__(self):
        # on Linux
        self.root_dir = '/home/steven/桌面/AICode/project_dataset/CK+/Emotion'
        self.r_dir_images = '/home/steven/桌面/AICode/project_dataset/CK+/cohn-kanade-images/'
        self.dirs_emotion_actors = self.get_dirs_emotion_actors()
        # actor means the code like S506,
        # every emotion dir is an actor's emotion dir, in which it's maybe more than one emotions
        # 找到每个演员所拥有的有效表情，也就是标签
        # 也找到每个表情所对应的演员
        # 找到每个图片所对应的标签
        self.dict_key_seq_labels = {}  # 字典,存储key值,value是标签值
        self.key_sequence_with_labels = []  # 存储 key值 (如S506/002),按这个key就可以索引出标签,图片,landmarks,而不会乱掉
        self.get_necessary_data()
        self.dict_actor_labels = {}  # key is actor s160, v is array including labels
        self.get_dict_actor_labels()
        self.dict_emotion_actors = {}  # key is emotion, v is array including acotrs
        self.get_dict_emotion_actors()
        self.dict_train_emotion_actors = {}  # train set
        self.dict_validate_emotion_actors = {}  # validation set
        self.whole_train_actors_list = []
        self.handle_train_validate_sets()

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
