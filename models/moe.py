import torch
import torch.nn as nn
from fmoe.transformer import FMoETransformerMLP
from fmoe.gates.faster_gate import FasterGate
from timm.models import vit_base_patch16_224, vit_small_patch16_224
import torch.nn.functional as F


class MoEModule(nn.Module):
    def __init__(self, num_new_experts, input_dim, hidden_dim, old_expert_blocks=[0, 11]):
        super(MoEModule, self).__init__()

        self.num_old_experts = len(old_expert_blocks)
        self.num_new_experts = num_new_experts
        self.old_expert_blocks = old_expert_blocks
        num_expert = self.num_new_experts + self.num_old_experts

        # self.proj_in = nn.Sequential(nn.Linear(input_dim, 768),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(768, hidden_dim)).cuda()
        self.moe = FMoETransformerMLP(num_expert, hidden_dim, hidden_dim, world_size=1, gate=FasterGate).cuda()
        # self.proj_out = nn.Sequential(nn.Linear(hidden_dim, 768),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(768, input_dim)).cuda()

        # load the pretrained layer to the old experts
        self.load_pretrained_weights()

    def load_pretrained_weights(self, device='cuda'):
        """
        load pretrained ViT ckpt
        old_expert_blocks: the selected block indexes list for the old experts
        """
        print(f'Load pretrained layer from: {self.old_expert_blocks}')
        load_dict = vit_base_patch16_224(pretrained=True).state_dict()

        attn_heads = [(load_dict[f'blocks.{i}.attn.qkv.weight'], load_dict[f'blocks.{i}.attn.qkv.bias'])
                        for i in self.old_expert_blocks]
        attn_projs = [(load_dict[f'blocks.{i}.attn.proj.weight'], load_dict[f'blocks.{i}.attn.proj.bias'])
                        for i in self.old_expert_blocks]

        # print(self.moe.experts[0].htoh4.weight.shape)
        # load the pretrained weight to old experts

        for i, (expert, attn_head) in enumerate(zip(self.moe.experts[:self.num_old_experts], attn_heads)):
            w_dim = attn_head[0].shape[0] // 3
            b_dim = attn_head[1].shape[0] // 3
            fc1_weight = attn_head[0][w_dim : w_dim * 2, :].unsqueeze(0).to(device)  # K
            fc1_bias = attn_head[1][b_dim : b_dim * 2].unsqueeze(0).to(device)  # K
            fc2_weight = attn_head[0][w_dim * 2 :, :].unsqueeze(0).to(device) # V
            fc2_bias = attn_head[1][b_dim * 2 :].unsqueeze(0).to(device) # V

            with torch.no_grad():
                expert.htoh4.weight = nn.Parameter(fc1_weight)
                expert.htoh4.bias = nn.Parameter(fc1_bias)
                expert.h4toh.weight = nn.Parameter(fc2_weight)
                expert.h4toh.bias = nn.Parameter(fc2_bias)

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.view(B, -1)
        # x = self.proj_in(x)
        x = self.moe(x)
        # x = self.proj_out(x)
        # x = x.view(B, N, C)
        return x

if __name__ == '__main__':
    num_old_experts = 2
    num_new_experts = 1
    expert_dim = 768
    input_dim = 768
    old_expert_blocks = [0, 11]

    x = torch.randn(100, 8, input_dim).cuda()  # 假设批量大小为10
    # 实例化MoEModule
    model = MoEModule(num_new_experts=1, input_dim=input_dim*8, hidden_dim=input_dim, old_expert_blocks=[0, 11])
    print(model)

    # 使用随机数据测试前向传播
    output = model(x)

    # 打印输出形状以手动检查
    print(f"Output shape: {output.shape}")