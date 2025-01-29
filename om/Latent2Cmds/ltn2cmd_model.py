import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class TransformerInverse1001(nn.Module):
    def __init__(self,
                encoder_input_size = 5,
                decoder_input_size = 16,
                output_size = 5, #predict future two Action
                nhead = 2     ,
                num_encoder_layers = 2  ,
                num_decoder_layers = 2  ,
                dim_feedforward = 512,
                mode = ''
                 ):
        super(TransformerInverse1001, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU()
        self.output_size = output_size
        self.encoder_embedding = nn.Linear(encoder_input_size, dim_feedforward)
        self.decoder_embedding = nn.Linear(decoder_input_size, dim_feedforward)

        # Positional Embeddings for sequence of length 2
        self.positional_embeddings = nn.Embedding(2, dim_feedforward)

        # Encoder
        encoder_layers = TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.pred_lmks_mlp0 = nn.Linear(dim_feedforward, dim_feedforward//2)
        self.pred_lmks_mlp1 = nn.Linear(dim_feedforward//2, dim_feedforward // 2)
        self.pred_lmks_mlp2 = nn.Linear(dim_feedforward // 2, decoder_input_size)

        # Decoder
        decoder_layers = TransformerDecoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward,batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        self.output_layer0 = nn.Linear(dim_feedforward, dim_feedforward//2)
        self.output_layer1 = nn.Linear(dim_feedforward//2, output_size)

    def freeze_encoder(self):
        """Freeze the encoder parameters."""
        for param in self.encoder_embedding.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the encoder parameters."""
        for param in self.encoder_embedding.parameters():
            param.requires_grad = True
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True

    def forward(self, encoder_input, decoder_input, current_epoch = 100):

        # Check the current epoch and freeze/unfreeze encoder accordingly
        # if current_epoch < 20:
        #     self.freeze_encoder()
        # else:
        #     self.unfreeze_encoder()

        # Generate position indices (0 and 1)
        position_indices = torch.arange(0, 2, dtype=torch.long, device=encoder_input.device)

        # Add positional embeddings
        encoder_input = self.encoder_embedding(encoder_input) + self.positional_embeddings(position_indices)

        # Get encoder output
        encoder_output = self.transformer_encoder(encoder_input)

        if self.mode == 'encoder':
            x = self.pred_lmks_mlp0(encoder_output)
            x = self.relu(x)
            x = self.pred_lmks_mlp1(x)
            x = self.relu(x)
            x = self.pred_lmks_mlp2(x)

            # Return encoder output directly
            return x
        else:
            decoder_input = self.decoder_embedding(decoder_input) + self.positional_embeddings(position_indices)
            # Pass through decoder
            output = self.transformer_decoder(decoder_input, encoder_output)
            output = self.relu(output)
            output = self.output_layer0(output)
            output = self.relu(output)
            output = self.output_layer1(output)

            return output


if __name__ == "__main__":

    # Example Usage
    encoder_input_size = 5
    decoder_input_size = 16
    output_size = 5
    nhead = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 512

    model = TransformerInverse1001(encoder_input_size, decoder_input_size, output_size, nhead, num_encoder_layers,
                              num_decoder_layers, dim_feedforward)

    # Adjust inputs for batch_first
    # For batch size = 1
    encoder_input = torch.rand(1, 2, 5)  # Example encoder input
    decoder_input = torch.rand(1, 2, 16)  # Example decoder input

    output = model(encoder_input, decoder_input)
    print(output)


    # d_input = 60 * 3 * 1 + 6 * 2
    # d_output = 6
    # model = inverse_model(input_size=d_input,label_size=d_output,d_hidden=2048)
    # model.eval()
    # with torch.no_grad():
    #     input_data = torch.ones((1, d_input))
    #     run_times = 1000
    #     t0 = time.time()
    #     for i in range(run_times):
    #         output_data = model.forward(input_data)
    #         # print(output_data.shape)
    #     t1 = time.time()
    #     print(1/((t1-t0)/run_times))
    #     print(((t1-t0)/run_times))
