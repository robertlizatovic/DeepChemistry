# packages
import pandas as pd
import numpy as np
import torch
import torch.utils.data as tud
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

# project modules
from utils import SMILESTokenizer, SMILESVocabulary, SMILESDataset, countTokens
from models import MolecularVAE

# util. functions
def compute_loss(model:nn.Module, batch:torch.Tensor) -> tuple:
    """Computes the NLL and KL loss for a batch of sequences"""
    batch_sz = batch.size(0)
    # prepare targets (input shifted 1 step to the left and padded with pad idx)
    targets = batch.roll(shifts=-1, dims=1)
    targets[:, -1] = model.pad_idx
    # prepare predictions
    logp, _, mean, logv = model(batch)
    preds = logp.permute(0, 2, 1)
    # compute NLL (neg. log likelihood) loss for the target sequence (ignore padding idx)
    nll_loss = F.nll_loss(preds, targets, ignore_index=model.pad_idx, reduction="sum") / batch_sz
    # compute KL loss
    kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp()) / batch_sz
    # compute total loss (averged within a batch)
    return (nll_loss, kl_loss)

def annealing_func(s:int, m:int, w:float) -> float:
    """
    Controls the final loss function through an adjustable KL weight
    -------------------
    s - iteration step
    m - midpoint (iteration step at which the KL weight is 0.5)
    w - width (smoothnes of the sigmoid annealing function around the midpoint)
    """
    return 1 / (1 + np.exp(-(s - m)/w))


if __name__ == "__main__":
    # load SMILES data
    NROWS = 10000
    chembl = pd.read_csv("data/cleaned_dataset.csv", nrows=NROWS)

    # create tokenizer/vocabulary
    tokenizer = SMILESTokenizer()
    vocabulary = SMILESVocabulary()

    # load vocabulary
    vocabulary.load("data/vocabulary.csv")

    # create train/test datasets
    train_ds = SMILESDataset(chembl["Smiles"][:int(0.8 * NROWS)], vocabulary, tokenizer)
    val_ds = SMILESDataset(chembl["Smiles"][int(0.8 * NROWS):], vocabulary, tokenizer)

    # create model
    vae = MolecularVAE(len(vocabulary), 32, 256, 256, vocabulary.getStartIdx(), vocabulary.getEndIdx(), vocabulary.getPadIdx(), use_unk_idx=True, enc_rnn_type="gru", 
        enc_rnn_layers=1, enc_bidirectional=True, enc_mid_layers=1, enc_mid_dim=256, enc_mid_dp=0.2, enc_mid_batchnorm=True, dec_mid_layers=1, dec_mid_dim=256, 
        dec_mid_dp=0.2, dec_mid_batchnorm=True, dec_rnn_type="gru", dec_rnn_layers=1, dec_token_dropout=0.2, dec_embedding_dp=0.0)
    print(vae)

    # num. of trainable params in the model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("# of trainable params: %i" % count_parameters(vae))

    # train the model
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    step = 0
    for e in range(1, 5):
        print("Epoch: %i" %e)
        dl = tud.DataLoader(train_ds, batch_size=128, shuffle=False, collate_fn=val_ds.getCollateFn())
        # enable dropout
        vae.train()
        for b, batch in enumerate(dl):
            optimizer.zero_grad()
            # compute loss
            nll_loss, kl_loss = compute_loss(vae, batch)
            kl_w = annealing_func(step, 250, 20)
            loss = nll_loss + kl_w * kl_loss
            # compute gradients
            loss.backward()
            # update weights
            optimizer.step()
            step += 1
            # print training progress
            print("Training batch: %i, ELBO: %.4f, NLL: %.4f, KL div: %.4f, KL weight: %.4f" % (b, 
                                loss.item(), nll_loss.item(), kl_loss.item(), kl_w))
        # evaluate on the validation set
        dl = tud.DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=val_ds.getCollateFn())
        # disable dropout
        vae.eval()
        val_scores = {"ELBO":[], "NLL":[], "KL":[]}
        for b, batch in enumerate(dl):
            nll_loss, kl_loss = compute_loss(vae, batch)
            loss = nll_loss + kl_loss
            val_scores["ELBO"].append(loss.item())
            val_scores["NLL"].append(nll_loss.item())
            val_scores["KL"].append(kl_loss.item())
        val_mean = {k:np.mean(v) for k, v in val_scores.items()}
        print("Epoch %i validation scores: ELBO: %.4f, NLL: %.4f, KL div: %.4f" % (e, 
                                    val_mean["ELBO"], val_mean["NLL"], val_mean["KL"]))


    # generate sequences
    vae.eval()
    samples = vae.generateSequences(n=32, max_len=150, greedy=False)
    gen_smiles = [tokenizer.untokenize(vocabulary.decode(s)) for s in samples.tolist()]
    print(gen_smiles)