import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    def __init__(self, d, k, v, num_head, r=23, num_pos=None):
        super(LambdaLayer, self).__init__()
        """ Multi-query Lambda layer (omitted intra-depth |u|, i.e., fixed as |u|=1)
        Args:
            - d: input channel
            - k: key/query depth
            - v: output channel (value depth)
            - num_head: number of head
            - m: context length
            - r: scope size
        """
        if (r is None and num_pos is None) or (r and num_pos):
            raise ValueError("One argument within ['r', 'num_pos'] must have a value, but not both")

        self.d, self.k, self.v, self.num_head, self.r, self.num_pos = d, k, v, num_head, r, num_pos

        self.query = nn.Sequential(
            nn.Conv2d(d, num_head*k, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_head*k)
        )
        self.key = nn.Conv2d(d, k, kernel_size=1, bias=False)

        self.value = nn.Sequential(
            nn.Conv2d(d, v, kernel_size=1, bias=False),
            nn.BatchNorm2d(v)
        )

        if r:
            self.lambda_conv = nn.Conv2d(1, k, kernel_size=(r,1), padding=((r-1)//2, 0))
        else:
            self.pos_embedding = nn.Parameter(torch.randn([num_pos, num_pos, k]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h*w

        query = self.query(x).reshape(b,self.num_head, self.k, n).transpose(2,3) # bhnk
        key = F.softmax(self.key(x).reshape(b, self.k, n), dim=-1).transpose(1,2) # bmk
        value = self.value(x).reshape(b, self.v, n).transpose(1,2) # bmv

        """ Create lambdas """
        lambda_c = torch.einsum('bmk, bmv -> bkv', key, value)
        if self.r:
            lambda_p = self.lambda_conv(value.reshape(b, 1, n, self.v)).transpose(1,2)
        else:
            lambda_p = torch.einsum('nmk, bmv -> bnkv',self.pos_embedding, value)

        lambda_final = lambda_c + lambda_p # bnkv
        return torch.einsum('bhnk, bnkv -> bnhv', query, lambda_final).reshape(b,h,w,-1).permute(0,3,1,2)

# if __name__ == '__main__':
#     sample = torch.randn([1,3,48,48])
#     lambda_layer = LambdaLayer(d=3, k=32, v=64, num_head=8, r=23, num_pos=None)
#     output = lambda_layer(sample)