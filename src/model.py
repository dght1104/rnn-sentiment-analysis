import torch
import torch.nn as nn
import torchtext.vocab as vocab

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=False):
        super().__init__()
        # Khởi tạo embedding layer (dùng GloVe nếu pretrained=True)
        if pretrained:
            glove = vocab.GloVe(name='6B', dim=embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(glove.vectors[:vocab_size])
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Khởi tạo khối RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Khởi tạo tầng Dense để dự đoán 3 nhãn
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Chuyển text thành embedding
        embedded = self.embedding(text)

        # Đưa qua khối RNN để lấy hidden state cuối
        output, hidden = self.rnn(embedded)

        # Đưa hidden state qua tầng Dense để dự đoán 3 nhãn
        out = self.fc(hidden.squeeze(0))

        return out

# Khởi tạo mô hình
model = RNNModel(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=True)