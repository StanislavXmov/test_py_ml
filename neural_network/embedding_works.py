from pathlib import Path

import torch
from navec import Navec

from embedding import WordsDataset, WordsRNN

ROOT = Path(__file__).resolve().parent.parent

navec = Navec.load(ROOT / "navec_hudlit_v1_12B_500K_300d_100q.tar")
d_train = WordsDataset(ROOT / "text", navec, prev_words=3)

model = WordsRNN(300, d_train.vocab_size)
model.load_state_dict(
    torch.load(ROOT / "model_rnn_words2.tar", map_location="cpu", weights_only=True)
)
model.eval()

predict = "отговорить вас от мечты".lower().split()
total = 10

for _ in range(total):
    _data = torch.vstack(
        [
            torch.tensor(d_train.navec_emb[predict[-x]])
            for x in range(d_train.prev_words, 0, -1)
        ]
    )
    p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1)
    predict.append(d_train.int_to_word[indx.item()])

print(" ".join(predict))
