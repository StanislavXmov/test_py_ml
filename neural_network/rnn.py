import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import re
from tqdm import tqdm


class TextRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 64
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.out(h)
        return y


class CharsDataset(data.Dataset):
    def __init__(self, path, prev_chars=3):
        self.prev_chars = prev_chars

        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()
            self.text = self.text.replace(
                "\ufeff", ""
            )  # убираем первый невидимый символ
            self.text = re.sub(
                r"[^А-яA-z0-9.,?;: ]", "", self.text
            )  # заменяем все неразрешенные символы на пустые символы

        self.text = self.text.lower()
        self.alphabet = set(self.text)
        self.int_to_alpha = dict(enumerate(sorted(self.alphabet)))
        self.alpha_to_int = {b: a for a, b in self.int_to_alpha.items()}
        self.num_characters = len(self.alphabet)
        self.onehots = torch.eye(self.num_characters)

    def __getitem__(self, item):
        _data = torch.vstack(
            [
                self.onehots[self.alpha_to_int[self.text[x]]]
                for x in range(item, item + self.prev_chars)
            ]
        )
        ch = self.text[item + self.prev_chars]
        t = self.alpha_to_int[ch]
        return _data, t

    def __len__(self):
        return len(self.text) - 1 - self.prev_chars


d_train = CharsDataset("train_data_true", prev_chars=10)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=False)

model = TextRNN(d_train.num_characters, d_train.num_characters)

optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

epochs = 100
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train).squeeze(0)
        loss = loss_func(predict, y_train.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(
            f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}"
        )

st = model.state_dict()
torch.save(st, "rnn_model.pth")


model.eval()
predict = "то увидите возможности".lower()
total = 40

for _ in range(total):
    _data = torch.vstack(
        [
            d_train.onehots[d_train.alpha_to_int[predict[-x]]]
            for x in range(d_train.prev_chars, 0, -1)
        ]
    )
    p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1)
    predict += d_train.int_to_alpha[indx.item()]

print(predict)
