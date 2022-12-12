import pickle
from dataclasses import dataclass

import torch
from torch import nn, optim, sqrt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from test_men import test_similarity_score
from util import Utils, get_arguments


@dataclass
class Config:
    jupyter_file = "glove_brown_model.pkl"
    y_max = 100
    alpha = 3 / 4
    path = "glove_brown.pt"

    @classmethod
    def get_path(cls, util_i: Utils):
        return util_i.file_prefix+cls.path

    @classmethod
    def get_best_path(cls, util_i: Utils):
        return util_i.file_prefix+"best_"+cls.path

    @classmethod
    def get_jupyter_file(cls, util_i: Utils):
        return util_i.file_prefix+cls.jupyter_file


class TransformedEmbedding(nn.Module):

    def __init__(self, word_embeddings):
        super().__init__()
        self.w_embed = nn.Embedding.from_pretrained(word_embeddings)
        self.embed_size = torch.tensor(word_embeddings.size(1))

        self.T = nn.Parameter(
            torch.rand(self.embed_size.item(), self.embed_size.item(), self.embed_size.item())
            * 2 * sqrt(3 / self.embed_size) - sqrt(3 / self.embed_size)
        )

        self.B = nn.Parameter(
            torch.zeros(self.embed_size.item(), self.embed_size.item())
        )

        self.time_vec = nn.Parameter(
            torch.rand(self.embed_size.item()) * 2 * sqrt(3 / self.embed_size) - sqrt(3 / self.embed_size)
        )

        self.op_layer = nn.Linear(self.embed_size.item(), self.embed_size.item())

    def forward(self, indices):

        word_embeddings = self.w_embed(indices)
        TransW = torch.einsum("bx,xyz->byz", word_embeddings, self.T) + self.B[None, :, :]
        h3 = torch.einsum("byz,z->by", TransW, self.time_vec)
        useW = self.op_layer(h3)

        return useW


class GloveModel(nn.Module):

    def __init__(self, word_embeddings):
        super().__init__()
        # self.w_center = nn.Embedding(word_embeddings.size(0), word_embeddings.size(1))
        # self.w_contex = nn.Embedding(word_embeddings.size(0), word_embeddings.size(1))

        self.w_center = TransformedEmbedding(word_embeddings)
        self.w_contex = TransformedEmbedding(word_embeddings)

        self.b_center = nn.Parameter(torch.zeros(word_embeddings.size(0)))
        self.b_contex = nn.Parameter(torch.zeros(word_embeddings.size(0)))

    def forward(self, indices):
        center_indices = indices[:, 0]
        context_indices = indices[:, 1]
        return torch.einsum("bi,bi->b", self.w_center(center_indices), self.w_contex(context_indices)) + \
            self.b_center[center_indices] + self.b_contex[context_indices]


def load_model(util_i: Utils):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GloveModel(util_i.get_glove_vectors())
    model = model.to(device)
    model.load_state_dict(torch.load(Config.get_path(util_i)))
    model = model.to(torch.device('cpu'))

    return model


def get_fy(y):
    fy = (y / Config.y_max)
    fy[fy > 1] = 1
    fy = fy ** Config.alpha

    return fy


def weighted_mse_loss(input_i, target, weight):
    return torch.sum(weight * (input_i - target) ** 2)


def train(X, y, util_i:Utils, epochs=100):
    model = GloveModel(util_i.get_glove_vectors())
    # model = load_model(util_i)

    word2index = util_i.get_word2index()

    fy = get_fy(y)
    log_y = torch.log(1 + y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)
    # optimizer = optim.Adagrad(model.parameters(), lr=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=5*1e-4)
    dataset = TensorDataset(X, fy, log_y)
    loader = DataLoader(dataset, batch_size=20_000, shuffle=True, num_workers=4)

    best_p_score = 0
    best_epoch = 0
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

            running_loss += loss.item()

        print(f"Epoch {t}: {running_loss}")

        word2vec = (model.w_center(torch.arange(len(word2index), dtype=torch.int64).to(device)) +
                    model.w_contex(torch.arange(len(word2index), dtype=torch.int64).to(device))) / 2
        word2vec = word2vec.detach().cpu().numpy()

        p_score = test_similarity_score(word2index, word2vec)

        prev_best_p_score = best_p_score
        if p_score > best_p_score:
            best_p_score = p_score
            best_epoch = t
            torch.save(model.state_dict(), Config.get_best_path(util_i))

        print("Curr Score:", p_score, "Best Score:", prev_best_p_score)
        print("Best Score Occurred in Epoch ", best_epoch)

    torch.save(model.state_dict(), Config.get_path(util_i))


def save_pkl_for_jupyter(util_i:Utils):
    word2index = util_i.get_word2index()

    model = load_model(util_i)

    word2vec = (model.w_center(torch.arange(len(word2index), dtype=torch.int64)) +
                model.w_contex(torch.arange(len(word2index), dtype=torch.int64))) / 2
    word2vec = word2vec.detach().numpy()

    with open(Config.get_jupyter_file(util_i), "wb") as f:
        pickle.dump((word2vec, word2index), f)


def similarity_score(util_i:Utils):
    word2index = util_i.get_word2index()

    model = load_model(util_i)

    word2vec = (model.w_center(torch.arange(len(word2index), dtype=torch.int64)) +
                model.w_contex(torch.arange(len(word2index), dtype=torch.int64))) / 2
    word2vec = word2vec.detach().numpy()

    test_similarity_score(word2index, word2vec)


if __name__ == "__main__":
    args = get_arguments()
    util_instance = Utils(context_size=args.c, context_weight=args.w)
    X, y = util_instance.get_data()
    train(X, y, util_instance)
    similarity_score(util_instance)
    save_pkl_for_jupyter(util_instance)
