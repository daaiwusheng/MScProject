from torchvision import models
import torch.nn as nn
import numpy as np

from  dataProcess.ckp_dataprovider import *

ckp_provider = CKPDataProvider()
# print(ckp_provider.dirs_emotion_actors)

# for k,v in ckp_provider.dict_emotion_actors.items():
#     print(v)
#     print(type(v))
#     print(len(v))


# print(ckp_provider.key_sequence_with_labels)
# print(len(ckp_provider.dict_emotion_actors))

print(ckp_provider.whole_train_actors_list)
print(len(ckp_provider.whole_train_actors_list))
print(len(set(ckp_provider.whole_train_actors_list)))
print(len(ckp_provider.dirs_emotion_actors))