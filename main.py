import gc
import re
from datetime import datetime
import chess
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from chessboard import display


def load_chess_data(elo, path):
    chess_data_raw = pd.read_csv(path)
    chess_data = chess_data_raw[chess_data_raw['WhiteElo'] > elo]
    del chess_data_raw
    gc.collect()
    chess_data = chess_data[['AN']]
    chess_data = chess_data[~chess_data['AN'].str.contains('{')]
    chess_data = chess_data[chess_data['AN'].str.len() > 20]
    print(chess_data.shape[0])

    return chess_data


def board_2_rep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep


def create_rep_layer(board, type):
    s = str(board)
    s = re.sub(f'[^{type}{type.upper()} \n]', '.', s)
    s = re.sub(f'{type}', '-1', s)
    s = re.sub(f'{type.upper()}', '1', s)
    s = re.sub(f'\.', '0', s)

    board_mat = []
    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        board_mat.append(row)

    return np.array(board_mat)


def move_2_rep(move, board):
    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8, 8))
    from_row = 8 - int(move[1])
    from_column = letter_2_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8, 8))
    to_row = 8 - int(move[3])
    to_column = letter_2_num[move[2]]
    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])


def create_move_list(s):
    return re.sub('\d*\. ', '', s).split(' ')[:-1]


class ChessDataset(Dataset):
    def __init__(self, games):
        super(ChessDataset, self).__init__()
        self.games = games

    def __len__(self):
        return 40_000

    def __getitem__(self, index):
        game_i = np.random.randint(self.games.shape[0])
        random_game = chess_data['AN'].values[game_i]
        moves = create_move_list(random_game)
        game_state_i = np.random.randint(len(moves) - 1)
        next_move = moves[game_state_i]
        moves = moves[:game_state_i]
        board = chess.Board()

        for move in moves:
            board.push_san(move)

        x = board_2_rep(board)
        y = move_2_rep(next_move, board)

        if game_state_i % 2 == 1:
            x *= -1

        return x, y


class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)

        return x


class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x


def check_mate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()


def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs ** 3
    probs = probs / probs.sum()

    return probs


def choose_move(board, color):
    legal_moves = list(board.legal_moves)
    move = check_mate_single(board)

    if move is not None:
        return move

    x = torch.Tensor(board_2_rep(board)).float().cuda()

    if color == 'BLACK':
        x *= -1

    x = x.unsqueeze(0)
    # print(x)
    move = predict(x)
    # print(move.shape)

    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))

    for from_ in froms:
        # val = move[0, :, :][8 - int(from_[1]), letter_2_num[from_[0]]]
        val = move[:, 0, 8 - int(from_[1]), letter_2_num[from_[0]]].cpu().detach().numpy()
        vals.append(val)

    probs = distribution_over_moves(vals)
    probs = probs.transpose()
    helper = []

    for item in probs:
        for iten in item:
            helper.append(iten)

    helper[np.argmax(helper)] += (1 - sum(helper))
    # print("sug1", legal_moves[np.argmax(helper)])
    chosen_move = legal_moves[np.argmax(helper)]

    # return chosen_move

    chosen_from = str(np.random.choice(froms, size=1, p=helper)[0])[:2]
    vals = []

    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]

        if from_ == chosen_from:
            to = str(legal_move)[2:]
            val = move[:, 1, 8 - int(to[1]), letter_2_num[to[0]]].cpu().detach().numpy()
            vals.append(val)
        else:
            vals.append(0)

    helper = []

    for item in vals:
        helper.append(float(item))

    # print(helper)
    # print("sug2", legal_moves[np.argmax(helper)])

    chosen_move = legal_moves[np.argmax(helper)]
    return chosen_move


def predict(x):
    return model(x)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(data_train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        inputs = inputs.to(torch.float32)

        # print(inputs)
        # print("HERE")
        # print(labels)
        # print(inputs.shape)
        outputs = model(inputs)
        # print(outputs)

        # Compute the loss and its gradients
        metric_from = nn.CrossEntropyLoss()
        metric_to = nn.CrossEntropyLoss()
        loss_from = metric_from(outputs[:, 0, :], labels[:, 0, :])
        loss_to = metric_to(outputs[:, 1, :], labels[:, 1, :])
        loss = loss_to + loss_from
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(data_train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train_more_epochs(epochs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/chess_bot_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(data_validate_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.cuda(), vlabels.cuda()
                vinputs = vinputs.to(torch.float32)
                voutputs = model(vinputs)
                metric_from = nn.CrossEntropyLoss()
                metric_to = nn.CrossEntropyLoss()
                loss_from = metric_from(voutputs[:, 0, :], vlabels[:, 0, :])
                loss_to = metric_to(voutputs[:, 1, :], vlabels[:, 1, :])
                loss = loss_to + loss_from
                running_vloss += loss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


def play():
    while True:
        display.check_for_quit()
        display.update(board.board_fen(), game_board)

        if game_board.flipped:
            display.flip(game_board)

        print(board)
        your_move = input()
        try:
            board.push_uci(your_move)
            print(board)
            print()
            move = choose_move(board, 'BLACK')
            board.push_uci(str(move))
            print(board)
            print()
        except chess.IllegalMoveError:
            print("Illegal move!")


letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

"""

chess_data = load_chess_data(2000, 'chess_archive/chess_games_train.csv')
data_train = ChessDataset(chess_data['AN'])
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True)

"""
"""

validation_data = load_chess_data(2000, 'chess_archive/chess_games_validate.csv')
vdata = ChessDataset(validation_data['AN'])
validation_data_loader = DataLoader(vdata, batch_size=32, shuffle=True, drop_last=True)

"""


"""

chess_data = load_chess_data(2000, 'chess_archive/chess_games_train.csv')
batch_size = 32
validation_split = .2
shuffle_dataset = True
random_seed = 42

dataset_size = chess_data.shape[0]
print(dataset_size)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

data_train = ChessDataset(chess_data['AN'])

data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, sampler=train_sampler)
data_validate_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, sampler=valid_sampler)


"""

"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ChessNet()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_more_epochs(75)

"""

#"""

model = ChessNet().cuda()
# model.load_state_dict(torch.load('models/model_20231101_203144_46'))
model.load_state_dict(torch.load('models/model_20231102_011510_15'))
board = chess.Board()
game_board = display.start()
display.flip(game_board)
play()

#"""