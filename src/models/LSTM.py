from torch import nn


# LSTM 模型
class LSTM(nn.Module):
    def __init__(self,
                 input_size=8,
                 hidden_size=32,
                 num_layers=1,
                 output_size=1,
                 dropout=0,
                 batch_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # x = x.view(x.shape[0], -1, 1)
        # None即隐层状态用0初始化 / 表示hidden state会用全0的state
        out, (h_n, h_c) = self.lstm(x, None)  # x -> out: [16, 5, 8] -> [16, 5, 32]
        # print(h_n.shape)
        out = self.fc(h_n)  # [2, 16, 32] -> [2, 16, 1]
        # out = self.fc(out[:, -1, :])  # 只需要最后一个的output [16, 32] -> [16, 1]
        return out
