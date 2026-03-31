# Requirements
# - PyTorch
# - Kagglehub
# - Pandas
# - Datasets

from datasets import load_dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import re
import torch
import numpy as np

def get_two_resp(convo):
        responses = re.split(" '[ \n]*' ", convo)

        if len(responses) >= 2:
            return responses[0], responses[1]
        else:
             return None, None
        
def clean_str(str):
     str = re.sub("[^ a-zA-Z]+", "", str)
     str = re.sub("[ ]+", " ", str)
     str = str.strip()
     str = str.lower()
     str = re.sub(" [ ]+", " ", str)
     return str
            

def clean(df):
    dialog = df["dialog"]
    
    q_a = np.empty((len(dialog), 2), dtype=np.dtypes.StringDType) # Numpy array where first col is question and second is the answer

    i = 0
    for convo in dialog:
        first_q, first_a = get_two_resp(convo)
        if first_q == None:
              continue

        clean_q = clean_str(first_q)
        clean_a = clean_str(first_a) 

        if len(clean_q) > 0 and len(clean_a) > 0:
            q_a[i][0] = clean_q
            q_a[i][1] = clean_a
            i = i + 1
        
    
    # Remove empty rows at the end
    q_a = q_a[0:i]

    return q_a



# Set the path to the file you'd like to load
file_path = "test.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "thedevastator/dailydialog-unlock-the-conversation-potential-in",
  file_path,
)




qa_dataset = clean(df)

# Tokenize the data

def validate_text(text):
    # Check for capital letters
    if bool(re.search(r'[A-Z]', text)):
        raise ValueError("Text must be all lowercase")
    if bool(re.search(r'[^a-z ]', text)):
        raise ValueError("Text must be all lowercase letters and spaces")
    if bool(re.search(r'  ', text)):
        raise ValueError("Text must not include double spaces")
    
    if len(text) <= 1:
        print(text)
        print("about to err")
    if text[0] == " " or text[len(text) - 1] == " ":
        print(text)
        print("is text")
        raise ValueError("Text must be stripped")


def load_token_dataset(dataset):
    start_token = 0
    end_token = 1
    token_map = {"PAD":0, "START_TOKEN":1, "END_TOKEN":2} # Converts token to id quickly
    id_map = ["PAD", "START_TOKEN", "END_TOKEN"] # Index is the token id. Converts id to string token


    token_id = 3 # Counter for new ID
    max_tokens = -1 # Track the most number of tokens needed for question or answer in the database
    for row in dataset:
        # Get the text from the question and answer
        questions = row[0]
        answers = row[1]
        validate_text(questions)
        validate_text(answers)

        question_strs = questions.split(" ")
        answer_strs = answers.split(" ")

        if len(question_strs) > max_tokens: 
            max_tokens = len(question_strs)
        if len(answer_strs) > max_tokens: 
            max_tokens = len(answer_strs)

        # Add each new unique question to token_map
        for t in question_strs:
            if token_map.get(t) == None:
                token_map[t] = token_id
                id_map.append(t)
                token_id = token_id + 1
        
        # Add each new unique answer to token_map
        for t in answer_strs:
            if token_map.get(t) == None:
                token_map[t] = token_id
                id_map.append(t)
                token_id = token_id + 1
    
    return token_map, id_map, max_tokens


# Tokenizes a string and pads with zeros to the longest sentence length plus two (for start/end tokens)
def tokenize(text, token_map, max_tokens_len):
    token_strs = text.lower().split(" ")
    if len(token_strs) > max_tokens_len + 2:
        raise ValueError("Text may not have more tokens than the max_tokens_len + 2")
    tokens = np.zeros(max_tokens_len + 2, dtype=np.int64) # Include two extra tokens for start and end
    tokens[0] = token_map["START_TOKEN"]
    i = 1
    for t in token_strs:
        if token_map.get(t) != None:
            tokens[i] = token_map.get(t)
            i = i + 1
        else:
            pass # Skip unknown inputs
        
    
    tokens[i] = token_map["END_TOKEN"]
    
    return np.array(tokens)

def tokenize_dataset(q_a, token_map, longest_sentence):
    rows, cols = np.shape(q_a)
    tokenized_inputs = np.zeros((rows, cols, longest_sentence + 2), dtype=np.int64) # Third dimen is longest_sentence + 2 to include start/end tokens

    for i, row in enumerate(q_a):
        tokenized_inputs[i][0] = tokenize(row[0], token_map, longest_sentence) # Returns a numpy array
        tokenized_inputs[i][1] = tokenize(row[1], token_map, longest_sentence) # Returns a numpy array

    return tokenized_inputs


token_map, id_map, max_tokens = load_token_dataset(qa_dataset)

tokenized_dataset = tokenize_dataset(qa_dataset, token_map, max_tokens)

MAX_TOKENS = max_tokens

# Convert tokens to tensors
tokenized_dataset = torch.from_numpy(tokenized_dataset)

# Positional encoding - gives the model the original order of inputs
# 
# The next block uses code from 

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) #for even positions using sin()
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) #for odd positions using cos()
    pos_encoding = angle_rads[np.newaxis,:]
    return pos_encoding

# Mask tokens. Any tokens beyond "END_TOKEN" will be masked
def create_padding_mask(seq):
    seq = torch.eq(torch.tensor([1,2,3,0,0]), 0).to(torch.int32)
    return seq[:, torch.newaxis, torch.newaxis, :]

# Mask future tokens
def create_lookahead_mask(size):
    return torch.triu(torch.ones((size, size)), 1)

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, torch.transpose(k)) 
    dk = np.shape(k)[-1].float()
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # -1e9 ~ (-INFINITY) => where ever mask is set, make its logit value close to -INF
    attention_weights = torch.nn.softmax(scaled_attention_logits, axis=-1)  
    output = torch.matmul(attention_weights, v)  

    return output, attention_weights

class MultiHeadAttentionClass(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.dense = torch.nn.Linear(d_model, d_model)
    
    # Resize into (batch_size, num_heads, seq_len, depth)
    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        x = torch.permute(x, (0, 2, 1, 3)) # Tf used transpose 
        return x

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = torch.permute(scaled_attention, (0, 2, 1, 3))
        concat_attention = torch.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return torch.nn.Sequential(torch.nn.Linear(d_model, dff), torch.nn.ReLU(), torch.nn.Linear(dff, d_model))

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttentionClass(d_model, num_heads)
        self.layerNorm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.layerNorm2 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
    
    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layerNorm1(x + attn_output)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layerNorm2(out1 + ffn_out)

        return out2

class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_positional_encoding, rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(input_vocab_size, max_positional_encoding)
        self.positional_encoding = positional_encoding(max_positional_encoding, d_model)
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for i in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask):
        sequence_length = x.shape[1]

        x = self.embedding(x)
        x *= torch.sqrt(self.d_model) # Done in research
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x)

        for i, el in enumerate(self.encoder_layers):
            x = el(x, mask)
        
        return x # (batch_size, input_seq_len, d_model)

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttentionClass(d_model, num_heads)
        self.mha2 = MultiHeadAttentionClass(d_model, num_heads)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

        self.layerNorm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.layerNorm2 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.layerNorm3 = torch.nn.LayerNorm(d_model, eps=1e-6)

        self.fnn = point_wise_feed_forward_network(d_model, dff)
    
    def forward(self, x, enc_layer, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layerNorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_layer, enc_layer, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layerNorm2(attn2 + out1)

        ffn_out = self.fnn(out2)
        ffn_out = self.dropout3(ffn_out)
        out3 = self.layerNorm3(ffn_out + out2)

        return out3, attn_weights_block1, attn_weights_block2



class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_positional_encoding, rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(target_vocab_size, max_positional_encoding)
        self.positional_encoding = positional_encoding(max_positional_encoding, d_model)
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for i in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, encoding_output, look_ahead_mask, padding_mask):
        sequence_length = x.shape[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= torch.sqrt(self.d_model)
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x)

        for i, dl in enumerate(self.decoder_layers):
            x, attn_weights_block1, attn_weights_block2 = dl(x, encoding_output, look_ahead_mask, padding_mask)
            attention_weights[['decoder_layer{}_block1'.format(i+1)]] = attn_weights_block1
            attention_weights[['decoder_layer{}_block2'.format(i+1)]] = attn_weights_block2
        
        return x, attention_weights

class Transformer(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.nn = torch.nn.Linear(d_model, target_vocab_size)
    
    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)
        dec_output, attn_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
        dec_output = self.nn(dec_output)

        return dec_output, attn_weights


