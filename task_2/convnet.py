import numpy as np
import torch
import base64
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class CdmcDataset(Dataset):
    def __init__(self, data, y=None, test=False):
        self.data = torch.Tensor(np.stack(data['ByteSequence'].apply(lambda x:
                                                                     [i for i in base64.b64decode(x)])))
        self.test = test
        if self.test:
            self.y = None
        else:
            self.y = torch.Tensor(y)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        input_vector = self.data[index]
        if self.test:
            return input_vector
        else:
            output_vector = self.y[index]
            return input_vector, output_vector


class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.Conv1 = nn.Conv1d(1, 64, 8)
        self.norm1 = nn.BatchNorm1d(64)
        self.Conv2 = nn.Conv1d(64, 128, 8)
        self.norm2 = nn.BatchNorm1d(128)
        self.Conv3 = nn.Conv1d(128, 256, 8)
        self.norm3 = nn.BatchNorm1d(256)
        self.Conv4 = nn.Conv1d(256, 512, 8)
        self.norm4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.LSTM = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.Linear2 = nn.Linear(512, 9)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, inputs):
        b_size = inputs.shape[0]
        X = self.Conv1(inputs)
        X = self.norm1(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.Conv2(X)
        X = self.norm2(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.Conv3(X)
        X = self.norm3(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.Conv4(X)
        X = self.norm4(X)
        X = self.relu(X)
        _, (hidden, _) = self.LSTM(X.reshape(b_size, -1, 512))
        hidden = hidden.view(1, 2, b_size, 256)[0]
        h1, h2 = hidden[0], hidden[1]
        X = torch.cat([h1, h2], axis=1)
        output = self.Linear2(X)
        return output


def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, batch_size, scheduler=None, split_num=1):
    loss_history = []
    train_history = []
    val_history = []
    max_f1 = 0.0
    batches_count = np.ceil(len(train_loader.dataset) / batch_size)

    for epoch in tqdm(range(num_epochs)):
        # with tqdm(total=batches_count) as progress_bar:
        model.train()  # Enter train mode

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            b_size = x.shape[0]
            prediction = model(x.reshape(b_size, 1, -1).to(device))
            loss_value = loss(prediction, y.argmax(axis=1).to(device))
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            _, indices = torch.max(prediction, 1)
            correct_batch_samples = torch.sum(indices == y.argmax(axis=1).to(device))
            correct_samples += correct_batch_samples
            total_samples += y.shape[0]

            loss_accum += loss_value

            # name = '[{} / {}] '.format(epoch + 1, num_epochs)
            # progress_bar.update()
            # progress_bar.set_description('{:>5s} Loss = {:.5f}, Acc = {:.3%}'.format(
            #     #                     name, loss.item(), accuracy)
            #     name, loss_accum / (i_step + 1), float(correct_batch_samples) / y.shape[0])
            # )

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy, f1 = compute_accuracy(model, val_loader)
        # if epoch > 30 and f1 > max_f1:
        #     max_f1 = f1
        #     torch.save(model, f'drive/My Drive/Colab Notebooks/model_{split_num}.pt')

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)

        if scheduler:
            scheduler.step()

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f, macro F1: %f" %
              (ave_loss, train_accuracy, val_accuracy, f1))

    return loss_history, train_history, val_history


def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
    model.eval()
    correct_samples = 0
    total_samples = 0
    true_labels = np.array([])
    preds = np.array([])

    for i_step, (x, y) in enumerate(loader):
        b_size = x.shape[0]
        prediction = model(x.reshape(b_size, 1, -1).to(device))
        # loss_value = loss(prediction, y.argmax(axis=1).to(device))

        _, indices = torch.max(prediction, 1)
        correct_samples += torch.sum(indices == y.argmax(axis=1).to(device))
        total_samples += y.shape[0]
        true_labels = np.hstack([true_labels, y.argmax(axis=1).numpy()])
        preds = np.hstack([preds, indices.cpu().numpy()])

    return float(correct_samples) / total_samples, f1_score(true_labels, preds, average='macro')


def nn_inference(model, loader, loss):
    model.eval()
    true_labels = np.array([])
    preds = np.array([])
    probs = None

    for i_step, (x, y) in tqdm(enumerate(loader)):
        b_size = x.shape[0]
        prediction = model(x.reshape(b_size, 1, -1).to(device))
        # loss_value = loss(prediction, y.argmax(axis=1))

        _, indices = torch.max(prediction, 1)
        true_labels = np.hstack([true_labels, y.argmax(axis=1).numpy()])
        preds = np.hstack([preds, indices.detach().cpu().numpy()])
        if type(probs) != type(None):
            probs = np.vstack([probs, prediction.detach().cpu().numpy()])
        else:
            probs = prediction.detach().cpu().numpy()
    print(f1_score(true_labels, preds, average='macro'))

    return true_labels, preds, probs


def nn_predict_test(model, loader):
    model.eval()
    preds = np.array([])
    probs = None

    for i_step, x in tqdm(enumerate(loader)):
        b_size = x.shape[0]
        prediction = model(x.reshape(b_size, 1, -1).to(device))

        _, indices = torch.max(prediction, 1)
        preds = np.hstack([preds, indices.detach().cpu().numpy()])
        if type(probs) != type(None):
            probs = np.vstack([probs, prediction.detach().cpu().numpy()])
        else:
            probs = prediction.detach().cpu().numpy()

    return preds, probs