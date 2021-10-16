import cv2
import torch
import sys
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score
from sklearn.metrics import recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt


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


# encoding=utf-8
def get_cur_info():
    file_name = sys._getframe().f_code.co_filename  # 当前文件名，可以通过__file__获得
    function_name = sys._getframe().f_code.co_name  # 当前函数名
    line_num = get_line_num()
    return file_name, function_name, line_num


def get_line_num():
    return sys._getframe().f_lineno  # 当前行号


def save_dict_as_csv(filename, dict_need_save):
    with open(filename, "w") as csv_file:
        # writer = csv.DictWriter(csv_file)
        writer = csv.writer(csv_file)
        for key, value in dict_need_save.items():
            writer.writerow([key, value])


def load_dict_from_csv(filename):
    read_dict = {}
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file)
        read_dict = dict(reader)
        return read_dict


def calculate_scores(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred, normalize=True)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    f_1_score = f1_score(y_true, y_pred, average=average)

    return acc, recall, precision, f_1_score


def draw_confusion_matrix(y_true, y_pred, labels):
    con_array = confusion_matrix(y_true, y_pred, labels)
    display = ConfusionMatrixDisplay(confusion_matrix=con_array, display_labels=labels)
    display.plot()
    plt.show()


Neutral = 0
Anger = 1
Contempt = 2
Disgust = 3
Fear = 4
Happiness = 5
Sadness = 6
Surprise = 7

emotion_labs_strings = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']


def emotions_to_str_names(emotions):
    emotions_string = []
    for i in range(len(emotions)):
        emotion = emotions[i]
        if emotion == Neutral:
            emotions_string.append('Neutral')
        elif emotion == 1:
            emotions_string.append('Anger')
        elif emotion == 2:
            emotions_string.append('Contempt')
        elif emotion == 3:
            emotions_string.append('Disgust')
        elif emotion == 4:
            emotions_string.append('Fear')
        elif emotion == 5:
            emotions_string.append('Happiness')
        elif emotion == 6:
            emotions_string.append('Sadness')
        elif emotion == 7:
            emotions_string.append('Surprise')

    return emotions_string
