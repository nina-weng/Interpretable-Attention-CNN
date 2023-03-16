import torch.nn as nn
import torch

class SelfAttentionBlock(nn.Module):
    def __init__(self,nb_features):
        super().__init__()

        self.nb_features = nb_features

        self.linear_k = nn.Linear(self.nb_features,self.nb_features)
        self.linear_v = nn.Linear(self.nb_features, self.nb_features)
        self.linear_q = nn.Linear(self.nb_features, self.nb_features)


    def forward(self,x): # shape of x: (batch_size, nb_features, nb_electrodes)
        x = torch.permute(x,(0,2,1)) # (batch_size, nb_electrodes, nb_features)

        k = self.linear_k(x)
        v = self.linear_v(x)
        q = self.linear_q(x)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask=None)
        out = scaled_attention
        out = torch.permute(out, (0, 2, 1))  #  (batch_size, nb_features, nb_electrodes)
        return out,attention_weights

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        # attn_dim: num_joints for spatial and seq_len for temporal
        '''
        The scaled dot product attention mechanism introduced in the Transformer
        :param q: the query vectors matrix (..., attn_dim, d_model/num_heads)
        :param k: the key vector matrix (..., attn_dim, d_model/num_heads)
        :param v: the value vector matrix (..., attn_dim, d_model/num_heads)
        :param mask: a mask for attention
        :return: the updated encoding and the attention weights matrix
        '''
        kt = k.transpose(-1, -2)
        matmul_qk = torch.matmul(q, kt)  # (..., num_heads, attn_dim, attn_dim)

        # scale matmul_qk
        dk = torch.tensor(k.shape[-1], dtype=torch.int32)  # tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = nn.functional.softmax(scaled_attention_logits,
                                                  dim=-1)  # (..., num_heads, attn_dim, attn_dim)

        attention_weights = torch.sum(attention_weights,dim=1)
        # get it from both dimensions
        # attention_weights = torch.sum(attention_weights, dim=1)+ torch.sum(attention_weights, dim=2)
        attention_weights = nn.functional.sigmoid(attention_weights)
        attention_weights = torch.unsqueeze(attention_weights,dim=-1)

        # output = torch.matmul(attention_weights, v)  # (..., num_heads, attn_dim, depth)
        output = v * attention_weights
        attention_weights = torch.squeeze(attention_weights,dim=-1)
        return output, attention_weights


class SEBlock(nn.Module):
    def __init__(self,nb_channels,nb_features,reduction_ratio=0.5):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.nb_features = nb_features
        self.nb_channels = nb_channels # number of electrodes in our cases
        self.middle_layer_param = int(self.nb_channels*self.reduction_ratio)

        self.global_pooling = nn.AvgPool1d(kernel_size=self.nb_features)
        self.fc_1 = nn.Linear(in_features=self.nb_channels,out_features=self.middle_layer_param)
        self.relu  = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=self.middle_layer_param,out_features=self.nb_channels)

    def forward(self,x):
        # shape of x: (batch_size, nb_features, nb_electrodes)
        permuted_x = torch.permute(x,(0,2,1)) # (batch_size, nb_electrodes, nb_features)
        out = self.global_pooling(permuted_x) # (batch_size, nb_electrodes, 1)
        out = torch.permute(out,(0,2,1)) # (batch_size, 1, nb_electrodes)

        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        scale = torch.sigmoid(out)  #(batch_size, 1, nb_electrodes)
        out = x * scale  # (batch_size, nb_features, nb_electrodes)

        scale = torch.squeeze(scale,dim=1) #(batch_size, nb_electrodes)

        return out, scale