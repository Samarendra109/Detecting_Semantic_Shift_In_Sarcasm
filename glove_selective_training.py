import pickle
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import glove_transformed
from test_men import test_similarity_score
from util import Utils, get_arguments


@dataclass
class Config:
    jupyter_file = "selective_glove_brown_model.pkl"
    y_max = 100
    alpha = 3 / 4
    path_transform = "glove_brown.pt"
    path = "selective_glove_brown.pt"

    @classmethod
    def get_path(cls, util_i: Utils):
        return util_i.file_prefix + cls.path

    @classmethod
    def get_best_path(cls, util_i: Utils):
        return util_i.file_prefix + "best_" + cls.path

    @classmethod
    def get_path_transform(cls, util_i: Utils):
        return util_i.file_prefix + cls.path_transform

    @classmethod
    def get_best_path_transform(cls, util_i: Utils):
        return util_i.file_prefix + "best_" + cls.path_transform

    @classmethod
    def get_jupyter_file(cls, util_i: Utils):
        return util_i.file_prefix + cls.jupyter_file


class GloveSelectiveModel(nn.Module):

    def __init__(self, w_center, b_center, w_contex, b_contex, train_indices):
        super().__init__()
        self.w_center = nn.Embedding.from_pretrained(w_center, freeze=False)
        self.w_contex = nn.Embedding.from_pretrained(w_contex, freeze=False)

        self.w_center_save = nn.Embedding.from_pretrained(w_center)
        self.w_contex_save = nn.Embedding.from_pretrained(w_contex)

        self.train_indices = train_indices  # I expect this to be a boolean array
        assert train_indices.size(0) == w_center.size(0)

        self.b_center = nn.Parameter(b_center)
        self.b_contex = nn.Parameter(b_contex)

        self.b_center_save = nn.Parameter(b_center)
        self.b_center_save.requires_grad = False
        self.b_contex_save = nn.Parameter(b_contex)
        self.b_contex_save.requires_grad = False

    def reset_selected_rows(self):
        with torch.no_grad():
            self.w_center.weight[~self.train_indices] = self.w_center_save.weight[~self.train_indices]
            self.w_contex.weight[~self.train_indices] = self.w_contex_save.weight[~self.train_indices]

            # self.b_center[~self.train_indices] = self.b_center_save[~self.train_indices]
            # self.b_contex[~self.train_indices] = self.b_contex_save[~self.train_indices]

    def forward(self, indices):
        center_indices = indices[:, 0]
        context_indices = indices[:, 1]
        return torch.einsum("bi,bi->b", self.w_center(center_indices), self.w_contex(context_indices)) + \
               self.b_center[center_indices] + self.b_contex[context_indices]


def load_model(util_i: Utils):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GloveSelectiveModel(*get_glove_vectors_and_training_indices(util_i))
    model = model.to(device)
    model.load_state_dict(torch.load(Config.get_best_path(util_i)))
    model = model.to(torch.device('cpu'))

    return model


def get_fy(y):
    fy = (y / Config.y_max)
    fy[fy > 1] = 1
    fy = fy ** Config.alpha

    return fy


def weighted_mse_loss(input_i, target, weight):
    return torch.sum(weight * (input_i - target) ** 2)


def train(X, y, util_i: Utils, epochs=100):
    model = GloveSelectiveModel(*get_glove_vectors_and_training_indices(util_i))
    # model = load_model(util_i)

    word2index = util_i.get_word2index()

    fy = get_fy(y)
    log_y = torch.log(1 + y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)
    # optimizer = optim.Adagrad(model.parameters(), lr=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=5 * 1e-4)
    print(model)
    dataset = TensorDataset(X, fy, log_y)
    loader = DataLoader(dataset, batch_size=20_000, shuffle=True, num_workers=4)

    best_p_score = 0
    best_epoch = 0
    best_loss = torch.inf
    for t in range(epochs):
        running_loss = 0.0
        for X_i, fy_i, log_y_i in tqdm(loader):
            X_i = X_i.to(device)
            fy_i = fy_i.to(device)
            log_y_i = log_y_i.to(device)

            optimizer.zero_grad()
            y_pred = model(X_i)
            loss = weighted_mse_loss(log_y_i, y_pred, fy_i)
            loss.backward()
            optimizer.step()
            model.reset_selected_rows()

            running_loss += loss.item()

        print(f"Epoch {t}: {running_loss}")

        word2vec = (model.w_center(torch.arange(len(word2index), dtype=torch.int64).to(device)) +
                    model.w_contex(torch.arange(len(word2index), dtype=torch.int64).to(device))) / 2
        word2vec = word2vec.detach().cpu().numpy()
        training_indices = model.train_indices.detach().cpu().numpy().nonzero()[0]

        word2vec_save = (model.w_center_save(torch.arange(len(word2index), dtype=torch.int64).to(device)) +
                         model.w_contex_save(torch.arange(len(word2index), dtype=torch.int64).to(device))) / 2
        word2vec_save = word2vec_save.detach().cpu().numpy()

        print((word2vec != word2vec_save).sum(axis=1).nonzero()[0])
        print(training_indices)

        #print(word2vec[13422])
        #print(word2vec_save[13422])

        p_score = test_similarity_score(word2index, word2vec, training_indices)

        prev_best_loss = best_loss
        if running_loss < best_loss:
            best_loss = running_loss
            best_epoch = t
            torch.save(model.state_dict(), Config.get_best_path(util_i))

        print("Curr Score:", running_loss, "Prev Best Score:", prev_best_loss)
        print("Best Score Occurred in Epoch ", best_epoch)

    torch.save(model.state_dict(), Config.get_path(util_i))


def get_glove_vectors_and_training_indices(util_i: Utils):
    word2index = util_i.get_word2index()
    training_indices = util_i.get_training_indices()
    # training_indices = torch.zeros(len(word2index), dtype=torch.bool)

    model = glove_transformed.load_model(util_i)

    word2vec_center = model.w_center(torch.arange(len(word2index), dtype=torch.int64)).detach().clone()
    word2vec_contex = model.w_contex(torch.arange(len(word2index), dtype=torch.int64)).detach().clone()

    bias_center = model.b_center.detach().data.clone()
    bias_contex = model.b_contex.detach().data.clone()

    return word2vec_center, bias_center, word2vec_contex, bias_contex, training_indices


def save_pkl_for_jupyter(util_i: Utils):
    word2index = util_i.get_word2index()

    model = load_model(util_i)

    word2vec = (model.w_center(torch.arange(len(word2index), dtype=torch.int64)) +
                model.w_contex(torch.arange(len(word2index), dtype=torch.int64))) / 2
    word2vec = word2vec.detach().numpy()

    word2vec_save = (model.w_center_save(torch.arange(len(word2index), dtype=torch.int64)) +
                     model.w_contex_save(torch.arange(len(word2index), dtype=torch.int64))) / 2
    word2vec_save = word2vec_save.detach().cpu().numpy()

    print((word2vec != word2vec_save).sum(axis=1).nonzero()[0])
    print(word2vec[209])
    print(word2vec_save[209])

    with open(Config.get_jupyter_file(util_i), "wb") as f:
        pickle.dump((word2vec, word2index), f)


def tmp_check(util_i):
    X, y = util_i.get_data()
    word2index = util_i.get_word2index()
    pass


if __name__ == "__main__":
    args = get_arguments()
    util_instance = Utils(context_size=args.c, context_weight=args.w)
    X, y = util_instance.get_data()
    train(X, y, util_instance)
    save_pkl_for_jupyter(util_instance)
