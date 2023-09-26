import numpy as np
import itertools
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

PRIME = 277149838960172320159769184791

def nums2str(nums):
    return [str(num) for num in nums]

def num2digits(num):
    num_digits = int(np.ceil(np.log10(num)))
    return [((num // 10**d) % 10) for d in range(num_digits)]

def space_num(num):
    return ' '.join(list(str(num)))

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.tokenizer.get_vocab_size()
    
    def decode(self, x, skip_special_tokens=False):
        out = self.tokenizer.decode(list(x), skip_special_tokens=skip_special_tokens)
        out = ''.join(out.split())
        return out
    
    def encode(self, x, return_tensors=None):
        out = self.tokenizer.encode(x.split(), is_pretokenized=True).ids
        if return_tensors is None:
            return out
        elif return_tensors == 'pt':
            return torch.tensor([out])
        return 

class ProductDataset(Dataset):

    def __init__(self, num_examples=10000, min_num=111, max_num=1000, max_length=None, split='train', joined_tokens=False):
        self.m = num_examples
        self.min_num = min_num
        self.max_num = max_num
        self.joined_tokens = joined_tokens
        self.split = split
        vocab = ['[UNK]', '[BOS]']
        vocab += [f'{i}' for i in range(10)]
        vocab += ['(', ')', '+', 'x', '=']
        if self.joined_tokens:
            vocab += [f'{i}x{j}' for i,j in itertools.product(range(10), range(10))]
        vocab = {v: i for i,v in enumerate(vocab)}
        self.tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        self.tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        self.max_length = max_length

    def __len__(self):
        return self.m
    
    def _is_train(self, num1, num2):
        count = (self.max_num-self.min_num)**2
        idx = (num1-self.min_num) + (self.max_num-self.min_num)*(num2-self.min_num)
        is_train = ((idx*PRIME % count) > count//4) # just some function
        return is_train


    def __getitem__(self, idx):
        found = False
        while not found:
            nums = np.random.randint(self.min_num, self.max_num, size=(10, 2))
            is_train = self._is_train(nums[:,0], nums[:,1])
            rel = is_train if self.split=='train' else ~is_train
            if np.any(rel):
                found = True
                nums = nums[rel]
                num1, num2 = nums[0]

        out_str = '[BOS] ' + self._nums_to_text(num1, num2)
        return {'input_ids': self.tokenizer_wrapper.encode(out_str)}

    def _nums_to_text(self, num1, num2):        
        digits1 = num2digits(num1)
        digits1 = [(digit, 10**d) for d,digit in enumerate(digits1)]
        digits2 = num2digits(num2)
        digits2 = [(digit, 10**d) for d,digit in enumerate(digits2)]
        out_str = f' {space_num(num1)} x {space_num(num2)} = '
        out_str += '( ' +' + '.join([f'{d} x {space_num(e)}' for d,e in digits1]) + ' ) x '
        out_str += '( ' +' + '.join([f'{d} x {space_num(e)}' for d,e in digits2]) + ' ) = '
        multiples = []
        products = []
        for (d1,e1), (d2,e2) in itertools.product(digits1, digits2):
            if self.joined_tokens:
                multi_str = f' {d1}x{d2} x {space_num(e1)} x {space_num(e2)} '
            else:
                multi_str = f' {d1} x {d2} x {space_num(e1)} x {space_num(e2)} '
            multiples.append(multi_str)
            prod = f'{d1*d2*e1*e2}'
            max_len = len(f'{e1*e2}')+1
            prod = '0'*(max_len-len(prod)) + prod
            products.append(space_num(prod))
        out_str += ' + '.join(multiples) + ' = '
        out_str += ' + '.join(products) + ' = '
        max_len = len(str((self.max_num-self.min_num)**2))
        prod = str(num1*num2)
        prod = '0'*(max_len-len(prod)) + prod
        out_str += ' ' + space_num(prod)
        return out_str


if __name__ == '__main__':
    
    i = 0
    ds = ProductDataset(min_num=1111, max_num=10000, max_length=165, joined_tokens=True)
    train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)
    
    for batch in tqdm(train_dataloader):
        pass
        print(batch)
        i+=1
        if i > 1:
            break