
import pandas as pd
import re
import torch
import torch.utils.data as tud

class SMILESTokenizer(object):
    
    def __init__(self, spec_tokens:list=["Br", "Cl"]):
        """
        Tokenizer that can convert a SMILES string to a list
        of tokens and vice versa.
        -----------------------
        spec_tokens - list of multi-character tokens (i.e. 'Br', 'Fe' etc.)
        """
        if (not spec_tokens) or (len(spec_tokens) == 0):
            self.pattern = None
        else:
            self.pattern = re.compile("({})".format("|".join(spec_tokens)))

    def getPattern(self):
        """Returns compiled regex pattern for multi-character tokens"""
        return self.pattern

    def tokenize(self, smi:str) -> list:
        """Tokenizes an input SMILES string"""
        if not self.pattern:
            return list(smi)
        # split input SMILES string using the supplied regex pattern    
        splitted = self.pattern.split(smi)
        tokens = []
        for i, s in enumerate(splitted):
            # make sure Br and Cl are treated as a single token
            if i % 2 == 0:
                tokens.extend(list(s))
            else:
                tokens.append(s)
        return tokens

    def untokenize(self, tokens:list) -> str:
        """Concatenates a list of tokens into a SMILES string"""
        return "".join(tokens)
    

class SMILESVocabulary(object):
    """Keeps track of string tokens and their associated integer indecies"""
    
    def __init__(self):
        self._tokens_idxs = {} # vocabulary
        self._current_idx = 0 # currently available index
        # add start, end, padding tokens
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.pad = "<pad>"
        self.update([self.pad, self.sos, self.eos])
            
    def __getitem__(self, token_or_idx):
        return self._tokens_idxs[token_or_idx]
    
    def __len__(self):
        return len(self._tokens_idxs) // 2
    
    def getStartIdx(self):
        return self._tokens_idxs[self.sos]
    
    def getEndIdx(self):
        return self._tokens_idxs[self.eos]
    
    def getPadIdx(self):
        return self._tokens_idxs[self.pad]
    
    def add(self, token:str):
        assert type(token) == str, "Token must be of type string."
        if token not in self._tokens_idxs:
            self._tokens_idxs[token] = self._current_idx
            self._tokens_idxs[self._current_idx] = token
            # update first available index
            self._current_idx += 1
            
    def tokens(self):
        """Returns a list of all tokens in the vocabulary"""
        return [t for t in self._tokens_idxs if type(t) == str]

    def update(self, tokens:list):
        """Updates the vocabulary with an iterable of tokens"""
        for t in tokens:
            self.add(t)
        
    def encode(self, tokens:list) -> list:
        """
        Encodes a list of tokens as a list of integer indecies.
        Attaches sos and eos idecies at the beginning and end of
        the encoded sequence.
        """
        smi_seq = [self._tokens_idxs[t] for t in tokens]
        return [self.getStartIdx()] + smi_seq + [self.getEndIdx()]
    
    def decode(self, indices:list) -> list:
        """
        Decodes a list of interger indices as a list of tokens.
        Ignores 1st sos idx and truncates the sequence if it encounters
        a special idx further down. 
        """
        tokens = []
        spec_idxs = [self.getStartIdx(), self.getEndIdx(), self.getPadIdx()]
        for i, idx in enumerate(indices):
            if (idx == self.getStartIdx()) & (i == 0):
                continue
            if (idx in spec_idxs) & (i > 0):
                break
            else:
                tokens.append(self._tokens_idxs[idx])
        return tokens

    def build(self, smiles:list, tokenizer:SMILESTokenizer) -> None:
        """
        Builds a vocabulary using a list of SMILES and
        an instance of a Tokenizer object. Any existing
        vocabulary is reset.
        -------------------------------
        smiles - iterable of SMILES strings
        tokenizer - instantiated SMILESTokenizer object
        
        returns None
        """
        # reset current vocabulary
        self.__init__()
        # build new vocabulary
        tokens = set()
        for smi in smiles:
            tokens.update(tokenizer.tokenize(smi))
        self.update(sorted(tokens))
        
    def save(self, path):
        """Saves the vocabulary to disk"""
        voc = [[k, v] for k, v in self._tokens_idxs.items() if isinstance(k, str)]
        voc_df = pd.DataFrame(voc, columns=["token", "index"])
        voc_df.to_csv(path, index=False)
    
    def load(self, path):
        """Loads a stored vocabulary from disk"""
        # reset current vocabulary
        self.__init__()
        # build vocabulary from csv file
        voc_df = pd.read_csv(path)
        for _, row in voc_df.iterrows():
            token = row["token"]
            idx = row["index"]
            self._tokens_idxs[token] = idx
            self._tokens_idxs[idx] = token
            self._current_idx = max(self._current_idx, idx)
        # update the currently available index
        self._current_idx += 1


class SMILESDataset(tud.Dataset):
    """Custom dataset class for producing batches of SMILES"""
    
    def __init__(self, smiles, vocabulary:SMILESVocabulary, tokenizer:SMILESTokenizer):
        """
        Creates a dataset from an iterable of SMILES, a built vocabulary of tokens
        and a SMILES tokenizer.
        """
        super().__init__()
        self._smiles = list(smiles)
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        
    def __getitem__(self, idx:int):
        """Returns a molecule at index idx as an encoded SMILES tensor"""
        smi = self._smiles[idx]
        smi_enc = self._vocabulary.encode(self._tokenizer.tokenize(smi))
        return torch.LongTensor(smi_enc)
    
    def __len__(self):
        return len(self._smiles)
    
    def getCollateFn(self):
        """
        Generates the collate function that will pad all
        sequences in a batch to the same max. length using the pad idx
        defined in the vocabulary.
        """
        def collate_fn(enc_tensors:list):
            """
            Pads encoded SMILES tensors to the same max. length using the pad idx.
            The output tensor has shape: (batch_sz, max. length)
            """
            batch_sz = len(enc_tensors)
            max_len = max([t.size(0) for t in enc_tensors])
            padded = torch.full((batch_sz, max_len), self._vocabulary.getPadIdx(), dtype=torch.long)
            # pad encoded batch of SMILES
            for i, t in enumerate(enc_tensors):
                padded[i, :t.size(0)] = t
            return padded
        
        return collate_fn


def countTokens(smiles:list, tokenizer:SMILESTokenizer, tokenCol:str="token", cntCol:str="cnt") -> pd.DataFrame:
    """Computes the token frequency in the smiles iterable"""
    token_cnts = {}
    for smi in smiles:
        # tokenize SMILES string
        tokenized = tokenizer.tokenize(smi)
        # count tokens
        for t in tokenized:
            try:
                token_cnts[t] += 1
            except KeyError:
                token_cnts[t] = 1
    return pd.DataFrame([[t, c] for t, c in token_cnts.items()], columns=[tokenCol, cntCol])
