from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F


class MolecularVAE(nn.Module):
    """
    Encodes a sequence as a probability distribution over a latent space - z
    and samples from this probability distribution.
    """
    
    def __init__(self, vocab_sz:int, embedding_dim:int, hidden_dim:int, latent_dim:int, sos_idx:int, eos_idx:int, pad_idx:int, 
        enc_rnn_type:str="gru", enc_bidirectional:bool=True, enc_rnn_layers:int=1, enc_mid_layers:int=0, enc_mid_dim:int=None, 
        enc_mid_dp:float=0.0, enc_mid_batchnorm:bool=False, dec_token_dropout:float=0.0, use_unk_idx:bool=False, dec_embedding_dp:float=0.0, 
        dec_mid_layers:int=0, dec_mid_dim:int=None, dec_mid_dp:float=0.0, dec_mid_batchnorm:bool=False, dec_rnn_type:str="gru", dec_rnn_layers:int=1):
        """Parameter initialization"""
        super().__init__()
        # general params
        self.vocab_sz = vocab_sz
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.use_unk_idx = use_unk_idx
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # encoder params
        self.enc_rnn_type = enc_rnn_type
        self.enc_bidirectional = enc_bidirectional
        self.enc_rnn_layers = enc_rnn_layers
        self.enc_hidden_factor = (2 if self.enc_bidirectional else 1) * self.enc_rnn_layers
        # decoder params
        self.dec_token_dropout = max(0.0, min(1.0, dec_token_dropout))
        self.dec_rnn_type = dec_rnn_type
        self.dec_rnn_layers = dec_rnn_layers
        
        # embedding layer (used by both encoder and decoder)
        # an extra embedding vector is used for the generic (unknown token). 
        # This token is never sampled by the output layer
        embedding_vocab_sz = self.vocab_sz + 1 if use_unk_idx else self.vocab_sz
        self.embedding = nn.Embedding(embedding_vocab_sz, self.embedding_dim, padding_idx=self.pad_idx)

        # encoder RNN (can be bidirectional in which case the output dim. is doubled)
        if self.enc_rnn_type == "gru":
            enc_rnn = nn.GRU
        elif self.enc_rnn_type == "lstm":
            enc_rnn = nn.LSTM
        else:
            raise ValueError("Invalid RNN type. Must be one of 'gru', 'lstm'")
        self.encoder_rnn = enc_rnn(self.embedding_dim, self.hidden_dim, num_layers=self.enc_rnn_layers,
                        batch_first=True, bidirectional=self.enc_bidirectional)
        
        # linear layers for computing the params of the latent vector z posterior distribution
        # (diagonal multivariate gaussian) from the hidden state vector of the RNN
        # build encoder layers from hidden to mean
        self.hidden2mean = self.build_middle_layers(enc_mid_layers, self.hidden_dim * self.enc_hidden_factor,
            enc_mid_dim, self.latent_dim, enc_mid_dp, enc_mid_batchnorm)
        # build encoder layers from hidden to logv
        self.hidden2logv = self.build_middle_layers(enc_mid_layers, self.hidden_dim * self.enc_hidden_factor,
            enc_mid_dim, self.latent_dim, enc_mid_dp, enc_mid_batchnorm)
        
        # linear layer for computing the decoder hidden vector input from the sampled latent vector
        self.latent2hidden = self.build_middle_layers(dec_mid_layers, self.latent_dim, dec_mid_dim, 
            self.hidden_dim * self.dec_rnn_layers, dec_mid_dp, dec_mid_batchnorm)
        
        # decoder layers
        self.embedding_droput = nn.Dropout(p=dec_embedding_dp)
        if self.dec_rnn_type == "gru":
            dec_rnn = nn.GRU
        elif self.dec_rnn_type == "lstm":
            dec_rnn = nn.LSTM
        else:
            raise ValueError("Invalid RNN type. Must be one of 'gru', 'lstm'")
        self.decoder_rnn = dec_rnn(self.embedding_dim, self.hidden_dim, num_layers=self.dec_rnn_layers, 
                  batch_first=True)
        self.outputs2vocab = nn.Linear(self.hidden_dim, self.vocab_sz)

    @staticmethod
    def build_middle_layers(n_layers:int, in_dim:int, mid_dim:int, out_dim:int, mid_dp:float, 
        mid_batchnorm:bool) -> nn.Module:
        """Creates a middle layer block"""
        if n_layers > 0:
            assert mid_dim is not None, "The dimension of the middle layer(s) must be provided."
            layers = []
            for i in range(n_layers):
                # stack middle layers
                if i == 0:
                    # first middle layer (in -> mid)
                    layers.append(nn.Linear(in_dim, mid_dim))
                else:
                    # subsequent middle layers (mid -> mid)
                    layers.append(nn.Linear(mid_dim, mid_dim))
                layers.append(nn.ReLU())
                if mid_dp > 0:
                    layers.append(nn.Dropout(p=mid_dp))
                if mid_batchnorm:
                    layers.append(nn.BatchNorm1d(mid_dim))
            # project last mid layer to output (mid -> out)
            layers.append(nn.Linear(mid_dim, out_dim))
            return nn.Sequential(*layers)
        else:
            return nn.Linear(in_dim, out_dim)
                
    def forward(self, input_seqs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the encoding and reparametrization step.
        ------------------------
        input_seqs: input batch of sequences (batch size, seq. length)
        
        returns (logp, z, mean, logv)
        """
        # run sequence through the encoder
        mean, logv, stdev = self.encode(input_seqs)
        # sample z from the posterior distribution
        z = self.samplePosterior(mean, stdev)
        # run through the decoder
        logp = self.decode(z, input_seqs, use_token_dropout=True)
        return logp, z, mean, logv
    
    def encode(self, input_seqs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes a sequence as parametrized posterior distribution over the latent space - z"""
        input_embeddings = self.embedding(input_seqs)
        if self.enc_rnn_type == "gru":
            _, hidden = self.encoder_rnn(input_embeddings)
        else:
            _, (hidden, _) = self.encoder_rnn(input_embeddings)
        # flatten RNN output
        hidden = hidden.view(-1, self.enc_hidden_factor * self.hidden_dim)
        # reparametrize (compute posterior distribution params)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        stdev = torch.exp(logv / 2)
        return mean, logv, stdev
        
    def decode(self, z:torch.Tensor, input_seqs:torch.Tensor, use_token_dropout:bool=True) -> torch.Tensor:
        """Decodes z as a log-likelihood over the vocabulary for each position in the sequence"""
        batch_sz = z.size(0)
        hidden = self.latent2hidden(z)
        hidden = hidden.view(self.dec_rnn_layers, batch_sz, self.hidden_dim)
        # prepare input sequences: if using token dropout, replace
        # some indices with dummy (padding) indices to weaken the decoder
        if use_token_dropout:
            input_seqs = input_seqs.clone()
            non_terminal_mask = (input_seqs != self.sos_idx) & (input_seqs != self.eos_idx)
            dp_mask = (torch.rand(input_seqs.size()) <= self.dec_token_dropout)
            dummy_idx = self.vocab_sz if self.use_unk_idx else self.pad_idx
            input_seqs[dp_mask & non_terminal_mask] = dummy_idx
        input_embeddings = self.embedding(input_seqs)
        input_embeddings = self.embedding_droput(input_embeddings)
        if self.dec_rnn_type == "gru":
            output, _ = self.decoder_rnn(input_embeddings, hidden)
        else:
            output, _ = self.decoder_rnn(input_embeddings, (hidden, torch.zeros_like(hidden)))
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
        if z is None:
            z = self.samplePrior(n)
        batch_sz = z.size(0)
        sequences = torch.full([batch_sz, max_len], self.pad_idx, dtype=torch.long)
        # set SOS idx at position 0 in the sequences
        sequences[:, 0] = self.sos_idx
        # running sequences
        running_mask = torch.ones(batch_sz).bool()
        for s in range(1, max_len):
            logp = self.decode(z[running_mask, :], sequences[running_mask, :s], use_token_dropout=False)
            # sample from log_softmax at sequence position - s
            next_idxs = self._sample(logp[: ,-1:, :], greedy=greedy).flatten()
            sequences[running_mask, s] = next_idxs
            # check for eos and pad signal and update running mask
            running_mask = (sequences[:, s] != self.eos_idx) & (sequences[:, s] != self.pad_idx) 
            if running_mask.sum() == 0:
                # all sequences are terminated
                break
        return sequences
            
    def _sample(self, logp:torch.Tensor, greedy:bool=False) -> torch.Tensor:
        """Samples idxs from a softmax distribution"""
        if greedy:
            # sample the most probable token at each sequence position
            return logp.argmax(-1)
        else:
            # randomly sample from a softmax distribution at each sequence position
            batch_sz, seq_len, vocab_sz = logp.size()
            rand = torch.rand(batch_sz, seq_len, 1).repeat(1, 1, vocab_sz)
            cdf = logp.exp().cumsum(-1)
            return (rand > cdf).long().sum(-1)

