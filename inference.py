import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import librosa
import pickle
import os
import torchaudio
from torch.utils.data import Dataset
import sys
import os
import copy
import torch
import torch.nn as nn
from conf.conf import Config
from model.NAW_models import NAW_LSTM_multi_view_model
from utils.steps import train
from utils.transforms import mel_spec_transform
from utils.utils import ensure_dir_exists

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from g2p_en import G2p
import time
g2p=G2p()
conf = Config()

transform = torch.nn.Sequential(
    mel_spec_transform(config=conf)
)


phones_dict={}
phones=open('data/lang/phones.txt').read().splitlines()
for i in range(len(phones)):
    key=phones[i].split()[0]
    value=int(phones[i].split()[1])
    phones_dict[key]=value


device = 'cpu'
model = NAW_LSTM_multi_view_model( input_dim_acoustic = conf.n_mels, input_dim_text = conf.input_dim_text , no_of_tokens = conf.no_of_tokens, hidden_dim = conf.hidden_dim, embedding_dim = conf.embedding_dim, num_layers=conf.num_layers, bidirectional=True).to(device)
model.load_state_dict(torch.load("exp/ckpt/dwd_0.1_clap_1/NAW_15.pt", map_location=torch.device(device)))

model.eval()


start = time.time()



waveform, sr = torchaudio.load('/home/ramesh720/Desktop/STD/small.wav')

keyword='keyword'
anc_phones = g2p(keyword.replace("'",""))
anc_seq=[phones_dict[x] for x in anc_phones]
anc_seq = np.array(anc_seq).astype('uint8')
anc_seq = torch.IntTensor(anc_seq).unsqueeze(0)
anc_length = torch.tensor(anc_seq.shape[0]).unsqueeze(0)
mel= transform(query_waveform).transpose(1,2)
mel_len = torch.tensor(mel.shape[1]).unsqueeze(0)

end =time.time()
print('time took', end-start)
query_embedding,query_embedding1,_ = model(mel.to(device),mel_len,anc_seq.to(device),anc_length)
sim_scores = torch.nn.functional.cosine_similarity(query_embedding, query_embedding1)

