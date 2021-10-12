import os.path

from torchvision import models
import torch.nn as nn
import numpy as np
from utility.tools import *




from  dataProcess.ckp_dataprovider import *


ckp_provider = CKPDataProvider(is_train=False)
print(len(ckp_provider))
# print(len(ckp_provider.actors_list))
# print(ckp_provider.dict_key_seq_labels)
# for k,v in ckp_provider.dict_train_image_dir_emotion.items():
#     print(k," ",v)
#
# print(len(ckp_provider.dict_validate_image_filename_emotion))

# print(len(ckp_provider.dict_emotion_actors))



# print(ckp_provider.whole_train_actors_list)
# print(len(ckp_provider.whole_train_actors_list))
# print(len(set(ckp_provider.whole_train_actors_list)))
# print(len(ckp_provider.dirs_emotion_actors))