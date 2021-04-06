import unittest
import torch
from models import MolecularVAE


class TestVAE(unittest.TestCase):

    VOCAB_SZ = 40
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 128
    LATENT_DIM = 256
    SOS_IDX = 1
    EOS_IDX = 2
    PAD_IDX = 0
    MID_DIM = 192

    def testAssembly(self):
        vae = MolecularVAE(self.VOCAB_SZ, self.EMBEDDING_DIM, self.HIDDEN_DIM, self.LATENT_DIM, self.SOS_IDX, 
            self.EOS_IDX, self.PAD_IDX, enc_rnn_type="gru",enc_bidirectional=True,
            enc_rnn_layers=1,enc_mid_layers=0, dec_token_dropout=0.5,use_unk_idx=False,
            dec_mid_layers=2,dec_rnn_type="gru", dec_rnn_layers=1)
        # dummy input
        batch = torch.randint(0, self.VOCAB_SZ, 16, 100, dtype=torch.long)
        pass


if __name__ == "__main__":
    unittest.main()