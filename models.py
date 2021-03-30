from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F


class MolecularVAE(nn.Module):
    """
    Encodes a sequence as a probability distribution over a latent space - z
    and samples from this probability distribution.
    """
    
    def __init__(self, vocab_sz:int, embedding_dim:int, hidden_dim:int, latent_dim:int, sos_idx:int, eos_idx:int, 
                 pad_idx:int, rnn_layers:int=1, bidirectional:bool=True, dropout:float=0.0):
        """Parameter initialization"""
        super().__init__()
        # module params
        self.vocab_sz = vocab_sz
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.rnn_layers = rnn_layers
        self.hidden_factor = (2 if self.bidirectional else 1) * self.rnn_layers
        
        # embedding layer (used by both encoder and decoder)
        self.embedding = nn.Embedding(self.vocab_sz, self.embedding_dim, padding_idx=self.pad_idx)
        
        # encoder RNN
        self.encoder_rnn = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.rnn_layers, 
                          batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
        
        # linear layers for computing the params of the latent vector z posterior distribution
        # (diagonal multivariate gaussian) from the hidden state vector of the RNN
        self.hidden2mean = nn.Linear(self.hidden_dim * self.hidden_factor, self.latent_dim)
        self.hidden2logv = nn.Linear(self.hidden_dim * self.hidden_factor, self.latent_dim)
        
        # linear layers for computing the decoder hidden vector from the latent vector
        self.latent2hidden = nn.Linear(self.latent_dim, self.hidden_dim * self.hidden_factor)
        
        # decoder layers
        self.decoder_rnn = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.rnn_layers, 
                  batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
        self.outputs2vocab = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.vocab_sz)
        
    def forward(self, input_seqs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the encoding and reparametrization step.
        ------------------------
        input_seqs: input batch of sequences (batch size, seq. length)
        
        returns (z, mean, logv)
        """
        input_embeddings = self.embedding(input_seqs)
        # run sequence through the encoder
        mean, logv, stdev = self.encode(input_embeddings)
        # sample z from the posterior distribution
        z = self.samplePosterior(mean, stdev)
        # run through the decoder
        logits = self.decode(z, input_embeddings)
        return logits, z, mean, logv
    
    def encode(self, input_embeddings:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes a sequence as parametrized posterior distribution over the latent space - z"""
        _, hidden = self.encoder_rnn(input_embeddings)
        # flatten RNN output
        hidden = hidden.view(-1, self.hidden_factor * self.hidden_dim)
        # reparametrize (compute posterior distribution params)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        stdev = torch.exp(logv / 2)
        return mean, logv, stdev
        
    def decode(self, z:torch.Tensor, input_embeddings:torch.Tensor) -> torch.Tensor:
        """Decodes z as a log-likelihood over the vocabulary for each position in the sequence"""
        batch_sz = z.size(0)
        hidden = self.latent2hidden(z)
        hidden = hidden.view(self.hidden_factor, batch_sz, self.hidden_dim)
        output, _ = self.decoder_rnn(input_embeddings, hidden)
        return F.log_softmax(self.outputs2vocab(output), dim=-1)
        
    def samplePrior(self, batch_sz:int) -> torch.Tensor:
        """Samples z from a unit multivariate Gaussian"""
        return torch.randn(batch_sz, self.latent_dim)

    def samplePosterior(self, mean:torch.Tensor, stdev:torch.Tensor) -> torch.Tensor:
        """
        Samples from the approximate multivariate Gaussian posterior parameterized by
        mean vector and diagonal covariance matrix.
        """
        batch_sz = mean.size(0)
        epsilon = self.samplePrior(batch_sz)
        return mean + stdev * epsilon
    
    def generateSequences(self, n:int=16, z=None, max_len:int=150, greedy:bool=False) -> torch.Tensor:
        """Generates a batch of sequences from latent space encodings."""
        # if z is not given, sample from prior
        if not z:
            z = self.samplePrior(n)
        batch_sz = z.size(0)
        sequences = torch.full([batch_sz, max_len], self.pad_idx, dtype=torch.long)
        # set SOS idx at position 0 in the sequences
        sequences[:, 0] = self.sos_idx
        # running embeddings
        input_embeddings = torch.zeros(batch_sz, max_len, self.embedding_dim)
        # running sequences
        running_mask = torch.ones(batch_sz).bool()
        for s in range(1, max_len):
            input_embeddings[running_mask, s-1, :] = self.embedding(sequences[running_mask, s-1])
            logits = self.decode(z[running_mask, :], input_embeddings[running_mask, :s, :])
            # sample from softmax at sequence position - s
            next_idxs = self._sample(logits[: ,-1:, :], greedy=greedy).flatten()
            sequences[running_mask, s] = next_idxs
            # check for eos and pad signal and update running mask
            running_mask = (sequences[:, s] != self.eos_idx) & (sequences[:, s] != self.pad_idx) 
            if running_mask.sum() == 0:
                # all sequences are terminated
                break
        return sequences
            
    def _sample(self, logits:torch.Tensor, greedy:bool=False) -> torch.Tensor:
        """Samples idxs from a softmax distribution"""
        if greedy:
            # sample the most probable token at each sequence position
            return logits.argmax(-1)
        else:
            # randomly sample from a softmax distribution at each sequence position
            batch_sz, seq_len, vocab_sz = logits.size()
            rand = torch.rand(batch_sz, seq_len, 1).repeat(1, 1, vocab_sz)
            cdf = logits.cumsum(-1)
            return (rand > cdf).long().sum(-1)

