import torch
import torch.nn as nn

class deffusion_module(nn.Module):
    def  __init__(self, model):
        super(deffusion_module, self).__init__()
        self.device = torch.device('cuda')
        self.arg0_1 = getattr(model.diffusion_conditioning.token_pair_proj, "0").weight
        self.arg1_1 = getattr(model.diffusion_conditioning.token_pair_proj, "0").bias
        self.arg2_1 = getattr(model.diffusion_conditioning.token_pair_proj, "1").weight
        self.arg3_1 = getattr(model.diffusion_conditioning.token_in_proj, "0").weight
        self.arg4_1 = getattr(model.diffusion_conditioning.token_in_proj, "0").bias
        self.arg5_1 = getattr(model.diffusion_conditioning.token_in_proj, "1").weight
        self.arg6_1 = model.diffusion_conditioning.single_trans1.layer_norm.weight
        self.arg7_1 = model.diffusion_conditioning.single_trans1.layer_norm.bias
        self.arg8_1 = model.diffusion_conditioning.single_trans1.linear_no_bias_ab.weight
        self.arg9_1 = model.diffusion_conditioning.single_trans1.linear_out.weight
        self.arg10_1 = model.diffusion_conditioning.pair_trans1.layer_norm.weight
        self.arg11_1 = model.diffusion_conditioning.pair_trans1.layer_norm.bias
        self.arg12_1 = model.diffusion_conditioning.pair_trans1.linear_no_bias_ab.weight
        self.arg13_1 = model.diffusion_conditioning.pair_trans1.linear_out.weight
        self.arg14_1 = model.diffusion_conditioning.single_trans2.layer_norm.weight
        self.arg15_1 = model.diffusion_conditioning.single_trans2.layer_norm.bias
        self.arg16_1 = model.diffusion_conditioning.single_trans2.linear_no_bias_ab.weight
        self.arg17_1 = model.diffusion_conditioning.single_trans2.linear_out.weight
        self.arg18_1 = model.diffusion_conditioning.pair_trans2.layer_norm.weight
        self.arg19_1 = model.diffusion_conditioning.pair_trans2.layer_norm.bias
        self.arg20_1 = model.diffusion_conditioning.pair_trans2.linear_no_bias_ab.weight
        self.arg21_1 = model.diffusion_conditioning.pair_trans2.linear_out.weight
        self.arg22_1 = model.diffusion_conditioning.fourier_embedding.weights
        self.arg23_1 = model.diffusion_conditioning.fourier_embedding.bias
        self.arg24_1 = getattr(model.diffusion_conditioning.fourier_proj, "0").weight
        self.arg25_1 = getattr(model.diffusion_conditioning.fourier_proj, "0").bias
        self.arg26_1 = getattr(model.diffusion_conditioning.fourier_proj, "1").weight
        self.arg27_1 = model.diffusion_conditioning.single_ln.weight
        self.arg28_1 = model.diffusion_conditioning.single_ln.bias
        self.arg29_1 = model.diffusion_conditioning.pair_ln.weight
        self.arg30_1 = model.diffusion_conditioning.pair_ln.bias
        self.arg31_1 = model.atom_attention_encoder.to_atom_cond.weight
        self.arg32_1 = getattr(model.atom_attention_encoder.token_to_atom_single, "0").weight
        self.arg33_1 = getattr(model.atom_attention_encoder.token_to_atom_single, "0").bias
        self.arg34_1 = getattr(model.atom_attention_encoder.token_to_atom_single, "1").weight
        self.arg35_1 = model.atom_attention_encoder.prev_pos_embed.weight
        self.arg36_1 = getattr(model.atom_attention_encoder.pair_update_block.atom_single_to_atom_pair_proj_h, "1").weight
        self.arg37_1 = getattr(model.atom_attention_encoder.pair_update_block.atom_single_to_atom_pair_proj_w, "1").weight
        self.arg38_1 = getattr(model.atom_attention_encoder.pair_update_block.atom_pair_mlp, "0").weight
        self.arg39_1 = getattr(model.atom_attention_encoder.pair_update_block.atom_pair_mlp, "2").weight
        self.arg40_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").ada_ln.lin_s_merged.weight
        self.arg41_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_a_nobias_double.weight
        self.arg42_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_b_nobias.weight
        self.arg43_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.weight
        self.arg44_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.bias
        self.arg45_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").ada_ln.lin_s_merged.weight
        self.arg46_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_a_nobias_double.weight
        self.arg47_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_b_nobias.weight
        self.arg48_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.weight
        self.arg49_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.bias
        self.arg50_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").ada_ln.lin_s_merged.weight
        self.arg51_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_a_nobias_double.weight
        self.arg52_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_b_nobias.weight
        self.arg53_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.weight
        self.arg54_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.bias
        self.arg55_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").q_bias
        self.arg56_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").single_layer_norm.lin_s_merged.weight
        self.arg57_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").to_qkv.weight
        self.arg58_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.weight
        self.arg59_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.bias
        self.arg60_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").q_bias
        self.arg61_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").single_layer_norm.lin_s_merged.weight
        self.arg62_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").to_qkv.weight
        self.arg63_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.weight
        self.arg64_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.bias
        self.arg65_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").q_bias
        self.arg66_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").single_layer_norm.lin_s_merged.weight
        self.arg67_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").to_qkv.weight
        self.arg68_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.weight
        self.arg69_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.bias
        self.arg70_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").weight
        self.arg71_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").bias
        self.arg72_1 = getattr(model.atom_attention_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "1").weight
        self.arg73_1 = getattr(model.atom_attention_encoder.to_token_single, "0").weight
        self.arg74_1 = getattr(model.atom_attention_encoder.token_pair_to_atom_pair, "0").weight
        self.arg75_1 = getattr(model.atom_attention_encoder.token_pair_to_atom_pair, "0").bias
        self.arg76_1 = getattr(model.atom_attention_encoder.token_pair_to_atom_pair, "1").weight
        self.arg77_1 = getattr(model.diffusion_transformer.blocks, "0").q_bias
        self.arg78_1 = getattr(model.diffusion_transformer.blocks, "0").transition.ada_ln.lin_s_merged.weight
        self.arg79_1 = getattr(model.diffusion_transformer.blocks, "0").transition.linear_a_nobias_double.weight
        self.arg80_1 = getattr(model.diffusion_transformer.blocks, "0").transition.linear_b_nobias.weight
        self.arg81_1 = getattr(model.diffusion_transformer.blocks, "0").transition.linear_s_biasinit_m2.weight
        self.arg82_1 = getattr(model.diffusion_transformer.blocks, "0").transition.linear_s_biasinit_m2.bias
        self.arg83_1 = getattr(model.diffusion_transformer.blocks, "0").norm_in.lin_s_merged.weight
        self.arg84_1 = getattr(model.diffusion_transformer.blocks, "0").to_qkv.weight
        self.arg85_1 = getattr(getattr(model.diffusion_transformer.blocks, "0").gate_proj, "0").weight
        self.arg86_1 = getattr(getattr(model.diffusion_transformer.blocks, "0").gate_proj, "0").bias
        self.arg87_1 = getattr(model.diffusion_transformer.blocks, "0").pair_layer_norm.weight
        self.arg88_1 = getattr(model.diffusion_transformer.blocks, "0").pair_layer_norm.bias
        self.arg89_1 = getattr(model.diffusion_transformer.blocks, "0").pair_linear.weight
        self.arg90_1 = getattr(model.diffusion_transformer.blocks, "0").to_out.weight
        self.arg91_1 = getattr(model.diffusion_transformer.blocks, "1").q_bias
        self.arg92_1 = getattr(model.diffusion_transformer.blocks, "1").transition.ada_ln.lin_s_merged.weight
        self.arg93_1 = getattr(model.diffusion_transformer.blocks, "1").transition.linear_a_nobias_double.weight
        self.arg94_1 = getattr(model.diffusion_transformer.blocks, "1").transition.linear_b_nobias.weight
        self.arg95_1 = getattr(model.diffusion_transformer.blocks, "1").transition.linear_s_biasinit_m2.weight
        self.arg96_1 = getattr(model.diffusion_transformer.blocks, "1").transition.linear_s_biasinit_m2.bias
        self.arg97_1 = getattr(model.diffusion_transformer.blocks, "1").norm_in.lin_s_merged.weight
        self.arg98_1 = getattr(model.diffusion_transformer.blocks, "1").to_qkv.weight
        self.arg99_1 = getattr(getattr(model.diffusion_transformer.blocks, "1").gate_proj, "0").weight
        self.arg100_1 = getattr(getattr(model.diffusion_transformer.blocks, "1").gate_proj, "0").bias
        self.arg101_1 = getattr(model.diffusion_transformer.blocks, "1").pair_layer_norm.weight
        self.arg102_1 = getattr(model.diffusion_transformer.blocks, "1").pair_layer_norm.bias
        self.arg103_1 = getattr(model.diffusion_transformer.blocks, "1").pair_linear.weight
        self.arg104_1 = getattr(model.diffusion_transformer.blocks, "1").to_out.weight
        self.arg105_1 = getattr(model.diffusion_transformer.blocks, "2").q_bias
        self.arg106_1 = getattr(model.diffusion_transformer.blocks, "2").transition.ada_ln.lin_s_merged.weight
        self.arg107_1 = getattr(model.diffusion_transformer.blocks, "2").transition.linear_a_nobias_double.weight
        self.arg108_1 = getattr(model.diffusion_transformer.blocks, "2").transition.linear_b_nobias.weight
        self.arg109_1 = getattr(model.diffusion_transformer.blocks, "2").transition.linear_s_biasinit_m2.weight
        self.arg110_1 = getattr(model.diffusion_transformer.blocks, "2").transition.linear_s_biasinit_m2.bias
        self.arg111_1 = getattr(model.diffusion_transformer.blocks, "2").norm_in.lin_s_merged.weight
        self.arg112_1 = getattr(model.diffusion_transformer.blocks, "2").to_qkv.weight
        self.arg113_1 = getattr(getattr(model.diffusion_transformer.blocks, "2").gate_proj, "0").weight
        self.arg114_1 = getattr(getattr(model.diffusion_transformer.blocks, "2").gate_proj, "0").bias
        self.arg115_1 = getattr(model.diffusion_transformer.blocks, "2").pair_layer_norm.weight
        self.arg116_1 = getattr(model.diffusion_transformer.blocks, "2").pair_layer_norm.bias
        self.arg117_1 = getattr(model.diffusion_transformer.blocks, "2").pair_linear.weight
        self.arg118_1 = getattr(model.diffusion_transformer.blocks, "2").to_out.weight
        self.arg119_1 = getattr(model.diffusion_transformer.blocks, "3").q_bias
        self.arg120_1 = getattr(model.diffusion_transformer.blocks, "3").transition.ada_ln.lin_s_merged.weight
        self.arg121_1 = getattr(model.diffusion_transformer.blocks, "3").transition.linear_a_nobias_double.weight
        self.arg122_1 = getattr(model.diffusion_transformer.blocks, "3").transition.linear_b_nobias.weight
        self.arg123_1 = getattr(model.diffusion_transformer.blocks, "3").transition.linear_s_biasinit_m2.weight
        self.arg124_1 = getattr(model.diffusion_transformer.blocks, "3").transition.linear_s_biasinit_m2.bias
        self.arg125_1 = getattr(model.diffusion_transformer.blocks, "3").norm_in.lin_s_merged.weight
        self.arg126_1 = getattr(model.diffusion_transformer.blocks, "3").to_qkv.weight
        self.arg127_1 = getattr(getattr(model.diffusion_transformer.blocks, "3").gate_proj, "0").weight
        self.arg128_1 = getattr(getattr(model.diffusion_transformer.blocks, "3").gate_proj, "0").bias
        self.arg129_1 = getattr(model.diffusion_transformer.blocks, "3").pair_layer_norm.weight
        self.arg130_1 = getattr(model.diffusion_transformer.blocks, "3").pair_layer_norm.bias
        self.arg131_1 = getattr(model.diffusion_transformer.blocks, "3").pair_linear.weight
        self.arg132_1 = getattr(model.diffusion_transformer.blocks, "3").to_out.weight
        self.arg133_1 = getattr(model.diffusion_transformer.blocks, "4").q_bias
        self.arg134_1 = getattr(model.diffusion_transformer.blocks, "4").transition.ada_ln.lin_s_merged.weight
        self.arg135_1 = getattr(model.diffusion_transformer.blocks, "4").transition.linear_a_nobias_double.weight
        self.arg136_1 = getattr(model.diffusion_transformer.blocks, "4").transition.linear_b_nobias.weight
        self.arg137_1 = getattr(model.diffusion_transformer.blocks, "4").transition.linear_s_biasinit_m2.weight
        self.arg138_1 = getattr(model.diffusion_transformer.blocks, "4").transition.linear_s_biasinit_m2.bias
        self.arg139_1 = getattr(model.diffusion_transformer.blocks, "4").norm_in.lin_s_merged.weight
        self.arg140_1 = getattr(model.diffusion_transformer.blocks, "4").to_qkv.weight
        self.arg141_1 = getattr(getattr(model.diffusion_transformer.blocks, "4").gate_proj, "0").weight
        self.arg142_1 = getattr(getattr(model.diffusion_transformer.blocks, "4").gate_proj, "0").bias
        self.arg143_1 = getattr(model.diffusion_transformer.blocks, "4").pair_layer_norm.weight
        self.arg144_1 = getattr(model.diffusion_transformer.blocks, "4").pair_layer_norm.bias
        self.arg145_1 = getattr(model.diffusion_transformer.blocks, "4").pair_linear.weight
        self.arg146_1 = getattr(model.diffusion_transformer.blocks, "4").to_out.weight
        self.arg147_1 = getattr(model.diffusion_transformer.blocks, "5").q_bias
        self.arg148_1 = getattr(model.diffusion_transformer.blocks, "5").transition.ada_ln.lin_s_merged.weight
        self.arg149_1 = getattr(model.diffusion_transformer.blocks, "5").transition.linear_a_nobias_double.weight
        self.arg150_1 = getattr(model.diffusion_transformer.blocks, "5").transition.linear_b_nobias.weight
        self.arg151_1 = getattr(model.diffusion_transformer.blocks, "5").transition.linear_s_biasinit_m2.weight
        self.arg152_1 = getattr(model.diffusion_transformer.blocks, "5").transition.linear_s_biasinit_m2.bias
        self.arg153_1 = getattr(model.diffusion_transformer.blocks, "5").norm_in.lin_s_merged.weight
        self.arg154_1 = getattr(model.diffusion_transformer.blocks, "5").to_qkv.weight
        self.arg155_1 = getattr(getattr(model.diffusion_transformer.blocks, "5").gate_proj, "0").weight
        self.arg156_1 = getattr(getattr(model.diffusion_transformer.blocks, "5").gate_proj, "0").bias
        self.arg157_1 = getattr(model.diffusion_transformer.blocks, "5").pair_layer_norm.weight
        self.arg158_1 = getattr(model.diffusion_transformer.blocks, "5").pair_layer_norm.bias
        self.arg159_1 = getattr(model.diffusion_transformer.blocks, "5").pair_linear.weight
        self.arg160_1 = getattr(model.diffusion_transformer.blocks, "5").to_out.weight
        self.arg161_1 = getattr(model.diffusion_transformer.blocks, "6").q_bias
        self.arg162_1 = getattr(model.diffusion_transformer.blocks, "6").transition.ada_ln.lin_s_merged.weight
        self.arg163_1 = getattr(model.diffusion_transformer.blocks, "6").transition.linear_a_nobias_double.weight
        self.arg164_1 = getattr(model.diffusion_transformer.blocks, "6").transition.linear_b_nobias.weight
        self.arg165_1 = getattr(model.diffusion_transformer.blocks, "6").transition.linear_s_biasinit_m2.weight
        self.arg166_1 = getattr(model.diffusion_transformer.blocks, "6").transition.linear_s_biasinit_m2.bias
        self.arg167_1 = getattr(model.diffusion_transformer.blocks, "6").norm_in.lin_s_merged.weight
        self.arg168_1 = getattr(model.diffusion_transformer.blocks, "6").to_qkv.weight
        self.arg169_1 = getattr(getattr(model.diffusion_transformer.blocks, "6").gate_proj, "0").weight
        self.arg170_1 = getattr(getattr(model.diffusion_transformer.blocks, "6").gate_proj, "0").bias
        self.arg171_1 = getattr(model.diffusion_transformer.blocks, "6").pair_layer_norm.weight
        self.arg172_1 = getattr(model.diffusion_transformer.blocks, "6").pair_layer_norm.bias
        self.arg173_1 = getattr(model.diffusion_transformer.blocks, "6").pair_linear.weight
        self.arg174_1 = getattr(model.diffusion_transformer.blocks, "6").to_out.weight
        self.arg175_1 = getattr(model.diffusion_transformer.blocks, "7").q_bias
        self.arg176_1 = getattr(model.diffusion_transformer.blocks, "7").transition.ada_ln.lin_s_merged.weight
        self.arg177_1 = getattr(model.diffusion_transformer.blocks, "7").transition.linear_a_nobias_double.weight
        self.arg178_1 = getattr(model.diffusion_transformer.blocks, "7").transition.linear_b_nobias.weight
        self.arg179_1 = getattr(model.diffusion_transformer.blocks, "7").transition.linear_s_biasinit_m2.weight
        self.arg180_1 = getattr(model.diffusion_transformer.blocks, "7").transition.linear_s_biasinit_m2.bias
        self.arg181_1 = getattr(model.diffusion_transformer.blocks, "7").norm_in.lin_s_merged.weight
        self.arg182_1 = getattr(model.diffusion_transformer.blocks, "7").to_qkv.weight
        self.arg183_1 = getattr(getattr(model.diffusion_transformer.blocks, "7").gate_proj, "0").weight
        self.arg184_1 = getattr(getattr(model.diffusion_transformer.blocks, "7").gate_proj, "0").bias
        self.arg185_1 = getattr(model.diffusion_transformer.blocks, "7").pair_layer_norm.weight
        self.arg186_1 = getattr(model.diffusion_transformer.blocks, "7").pair_layer_norm.bias
        self.arg187_1 = getattr(model.diffusion_transformer.blocks, "7").pair_linear.weight
        self.arg188_1 = getattr(model.diffusion_transformer.blocks, "7").to_out.weight
        self.arg189_1 = getattr(model.diffusion_transformer.blocks, "8").q_bias
        self.arg190_1 = getattr(model.diffusion_transformer.blocks, "8").transition.ada_ln.lin_s_merged.weight
        self.arg191_1 = getattr(model.diffusion_transformer.blocks, "8").transition.linear_a_nobias_double.weight
        self.arg192_1 = getattr(model.diffusion_transformer.blocks, "8").transition.linear_b_nobias.weight
        self.arg193_1 = getattr(model.diffusion_transformer.blocks, "8").transition.linear_s_biasinit_m2.weight
        self.arg194_1 = getattr(model.diffusion_transformer.blocks, "8").transition.linear_s_biasinit_m2.bias
        self.arg195_1 = getattr(model.diffusion_transformer.blocks, "8").norm_in.lin_s_merged.weight
        self.arg196_1 = getattr(model.diffusion_transformer.blocks, "8").to_qkv.weight
        self.arg197_1 = getattr(getattr(model.diffusion_transformer.blocks, "8").gate_proj, "0").weight
        self.arg198_1 = getattr(getattr(model.diffusion_transformer.blocks, "8").gate_proj, "0").bias
        self.arg199_1 = getattr(model.diffusion_transformer.blocks, "8").pair_layer_norm.weight
        self.arg200_1 = getattr(model.diffusion_transformer.blocks, "8").pair_layer_norm.bias
        self.arg201_1 = getattr(model.diffusion_transformer.blocks, "8").pair_linear.weight
        self.arg202_1 = getattr(model.diffusion_transformer.blocks, "8").to_out.weight
        self.arg203_1 = getattr(model.diffusion_transformer.blocks, "9").q_bias
        self.arg204_1 = getattr(model.diffusion_transformer.blocks, "9").transition.ada_ln.lin_s_merged.weight
        self.arg205_1 = getattr(model.diffusion_transformer.blocks, "9").transition.linear_a_nobias_double.weight
        self.arg206_1 = getattr(model.diffusion_transformer.blocks, "9").transition.linear_b_nobias.weight
        self.arg207_1 = getattr(model.diffusion_transformer.blocks, "9").transition.linear_s_biasinit_m2.weight
        self.arg208_1 = getattr(model.diffusion_transformer.blocks, "9").transition.linear_s_biasinit_m2.bias
        self.arg209_1 = getattr(model.diffusion_transformer.blocks, "9").norm_in.lin_s_merged.weight
        self.arg210_1 = getattr(model.diffusion_transformer.blocks, "9").to_qkv.weight
        self.arg211_1 = getattr(getattr(model.diffusion_transformer.blocks, "9").gate_proj, "0").weight
        self.arg212_1 = getattr(getattr(model.diffusion_transformer.blocks, "9").gate_proj, "0").bias
        self.arg213_1 = getattr(model.diffusion_transformer.blocks, "9").pair_layer_norm.weight
        self.arg214_1 = getattr(model.diffusion_transformer.blocks, "9").pair_layer_norm.bias
        self.arg215_1 = getattr(model.diffusion_transformer.blocks, "9").pair_linear.weight
        self.arg216_1 = getattr(model.diffusion_transformer.blocks, "9").to_out.weight
        self.arg217_1 = getattr(model.diffusion_transformer.blocks, "10").q_bias
        self.arg218_1 = getattr(model.diffusion_transformer.blocks, "10").transition.ada_ln.lin_s_merged.weight
        self.arg219_1 = getattr(model.diffusion_transformer.blocks, "10").transition.linear_a_nobias_double.weight
        self.arg220_1 = getattr(model.diffusion_transformer.blocks, "10").transition.linear_b_nobias.weight
        self.arg221_1 = getattr(model.diffusion_transformer.blocks, "10").transition.linear_s_biasinit_m2.weight
        self.arg222_1 = getattr(model.diffusion_transformer.blocks, "10").transition.linear_s_biasinit_m2.bias
        self.arg223_1 = getattr(model.diffusion_transformer.blocks, "10").norm_in.lin_s_merged.weight
        self.arg224_1 = getattr(model.diffusion_transformer.blocks, "10").to_qkv.weight
        self.arg225_1 = getattr(getattr(model.diffusion_transformer.blocks, "10").gate_proj, "0").weight
        self.arg226_1 = getattr(getattr(model.diffusion_transformer.blocks, "10").gate_proj, "0").bias
        self.arg227_1 = getattr(model.diffusion_transformer.blocks, "10").pair_layer_norm.weight
        self.arg228_1 = getattr(model.diffusion_transformer.blocks, "10").pair_layer_norm.bias
        self.arg229_1 = getattr(model.diffusion_transformer.blocks, "10").pair_linear.weight
        self.arg230_1 = getattr(model.diffusion_transformer.blocks, "10").to_out.weight
        self.arg231_1 = getattr(model.diffusion_transformer.blocks, "11").q_bias
        self.arg232_1 = getattr(model.diffusion_transformer.blocks, "11").transition.ada_ln.lin_s_merged.weight
        self.arg233_1 = getattr(model.diffusion_transformer.blocks, "11").transition.linear_a_nobias_double.weight
        self.arg234_1 = getattr(model.diffusion_transformer.blocks, "11").transition.linear_b_nobias.weight
        self.arg235_1 = getattr(model.diffusion_transformer.blocks, "11").transition.linear_s_biasinit_m2.weight
        self.arg236_1 = getattr(model.diffusion_transformer.blocks, "11").transition.linear_s_biasinit_m2.bias
        self.arg237_1 = getattr(model.diffusion_transformer.blocks, "11").norm_in.lin_s_merged.weight
        self.arg238_1 = getattr(model.diffusion_transformer.blocks, "11").to_qkv.weight
        self.arg239_1 = getattr(getattr(model.diffusion_transformer.blocks, "11").gate_proj, "0").weight
        self.arg240_1 = getattr(getattr(model.diffusion_transformer.blocks, "11").gate_proj, "0").bias
        self.arg241_1 = getattr(model.diffusion_transformer.blocks, "11").pair_layer_norm.weight
        self.arg242_1 = getattr(model.diffusion_transformer.blocks, "11").pair_layer_norm.bias
        self.arg243_1 = getattr(model.diffusion_transformer.blocks, "11").pair_linear.weight
        self.arg244_1 = getattr(model.diffusion_transformer.blocks, "11").to_out.weight
        self.arg245_1 = getattr(model.diffusion_transformer.blocks, "12").q_bias
        self.arg246_1 = getattr(model.diffusion_transformer.blocks, "12").transition.ada_ln.lin_s_merged.weight
        self.arg247_1 = getattr(model.diffusion_transformer.blocks, "12").transition.linear_a_nobias_double.weight
        self.arg248_1 = getattr(model.diffusion_transformer.blocks, "12").transition.linear_b_nobias.weight
        self.arg249_1 = getattr(model.diffusion_transformer.blocks, "12").transition.linear_s_biasinit_m2.weight
        self.arg250_1 = getattr(model.diffusion_transformer.blocks, "12").transition.linear_s_biasinit_m2.bias
        self.arg251_1 = getattr(model.diffusion_transformer.blocks, "12").norm_in.lin_s_merged.weight
        self.arg252_1 = getattr(model.diffusion_transformer.blocks, "12").to_qkv.weight
        self.arg253_1 = getattr(getattr(model.diffusion_transformer.blocks, "12").gate_proj, "0").weight
        self.arg254_1 = getattr(getattr(model.diffusion_transformer.blocks, "12").gate_proj, "0").bias
        self.arg255_1 = getattr(model.diffusion_transformer.blocks, "12").pair_layer_norm.weight
        self.arg256_1 = getattr(model.diffusion_transformer.blocks, "12").pair_layer_norm.bias
        self.arg257_1 = getattr(model.diffusion_transformer.blocks, "12").pair_linear.weight
        self.arg258_1 = getattr(model.diffusion_transformer.blocks, "12").to_out.weight
        self.arg259_1 = getattr(model.diffusion_transformer.blocks, "13").q_bias
        self.arg260_1 = getattr(model.diffusion_transformer.blocks, "13").transition.ada_ln.lin_s_merged.weight
        self.arg261_1 = getattr(model.diffusion_transformer.blocks, "13").transition.linear_a_nobias_double.weight
        self.arg262_1 = getattr(model.diffusion_transformer.blocks, "13").transition.linear_b_nobias.weight
        self.arg263_1 = getattr(model.diffusion_transformer.blocks, "13").transition.linear_s_biasinit_m2.weight
        self.arg264_1 = getattr(model.diffusion_transformer.blocks, "13").transition.linear_s_biasinit_m2.bias
        self.arg265_1 = getattr(model.diffusion_transformer.blocks, "13").norm_in.lin_s_merged.weight
        self.arg266_1 = getattr(model.diffusion_transformer.blocks, "13").to_qkv.weight
        self.arg267_1 = getattr(getattr(model.diffusion_transformer.blocks, "13").gate_proj, "0").weight
        self.arg268_1 = getattr(getattr(model.diffusion_transformer.blocks, "13").gate_proj, "0").bias
        self.arg269_1 = getattr(model.diffusion_transformer.blocks, "13").pair_layer_norm.weight
        self.arg270_1 = getattr(model.diffusion_transformer.blocks, "13").pair_layer_norm.bias
        self.arg271_1 = getattr(model.diffusion_transformer.blocks, "13").pair_linear.weight
        self.arg272_1 = getattr(model.diffusion_transformer.blocks, "13").to_out.weight
        self.arg273_1 = getattr(model.diffusion_transformer.blocks, "14").q_bias
        self.arg274_1 = getattr(model.diffusion_transformer.blocks, "14").transition.ada_ln.lin_s_merged.weight
        self.arg275_1 = getattr(model.diffusion_transformer.blocks, "14").transition.linear_a_nobias_double.weight
        self.arg276_1 = getattr(model.diffusion_transformer.blocks, "14").transition.linear_b_nobias.weight
        self.arg277_1 = getattr(model.diffusion_transformer.blocks, "14").transition.linear_s_biasinit_m2.weight
        self.arg278_1 = getattr(model.diffusion_transformer.blocks, "14").transition.linear_s_biasinit_m2.bias
        self.arg279_1 = getattr(model.diffusion_transformer.blocks, "14").norm_in.lin_s_merged.weight
        self.arg280_1 = getattr(model.diffusion_transformer.blocks, "14").to_qkv.weight
        self.arg281_1 = getattr(getattr(model.diffusion_transformer.blocks, "14").gate_proj, "0").weight
        self.arg282_1 = getattr(getattr(model.diffusion_transformer.blocks, "14").gate_proj, "0").bias
        self.arg283_1 = getattr(model.diffusion_transformer.blocks, "14").pair_layer_norm.weight
        self.arg284_1 = getattr(model.diffusion_transformer.blocks, "14").pair_layer_norm.bias
        self.arg285_1 = getattr(model.diffusion_transformer.blocks, "14").pair_linear.weight
        self.arg286_1 = getattr(model.diffusion_transformer.blocks, "14").to_out.weight
        self.arg287_1 = getattr(model.diffusion_transformer.blocks, "15").q_bias
        self.arg288_1 = getattr(model.diffusion_transformer.blocks, "15").transition.ada_ln.lin_s_merged.weight
        self.arg289_1 = getattr(model.diffusion_transformer.blocks, "15").transition.linear_a_nobias_double.weight
        self.arg290_1 = getattr(model.diffusion_transformer.blocks, "15").transition.linear_b_nobias.weight
        self.arg291_1 = getattr(model.diffusion_transformer.blocks, "15").transition.linear_s_biasinit_m2.weight
        self.arg292_1 = getattr(model.diffusion_transformer.blocks, "15").transition.linear_s_biasinit_m2.bias
        self.arg293_1 = getattr(model.diffusion_transformer.blocks, "15").norm_in.lin_s_merged.weight
        self.arg294_1 = getattr(model.diffusion_transformer.blocks, "15").to_qkv.weight
        self.arg295_1 = getattr(getattr(model.diffusion_transformer.blocks, "15").gate_proj, "0").weight
        self.arg296_1 = getattr(getattr(model.diffusion_transformer.blocks, "15").gate_proj, "0").bias
        self.arg297_1 = getattr(model.diffusion_transformer.blocks, "15").pair_layer_norm.weight
        self.arg298_1 = getattr(model.diffusion_transformer.blocks, "15").pair_layer_norm.bias
        self.arg299_1 = getattr(model.diffusion_transformer.blocks, "15").pair_linear.weight
        self.arg300_1 = getattr(model.diffusion_transformer.blocks, "15").to_out.weight
        self.arg301_1 = model.atom_attention_decoder.token_to_atom.weight
        self.arg302_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").ada_ln.lin_s_merged.weight
        self.arg303_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_a_nobias_double.weight
        self.arg304_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_b_nobias.weight
        self.arg305_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.weight
        self.arg306_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.bias
        self.arg307_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").ada_ln.lin_s_merged.weight
        self.arg308_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_a_nobias_double.weight
        self.arg309_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_b_nobias.weight
        self.arg310_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.weight
        self.arg311_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.bias
        self.arg312_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").ada_ln.lin_s_merged.weight
        self.arg313_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_a_nobias_double.weight
        self.arg314_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_b_nobias.weight
        self.arg315_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.weight
        self.arg316_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.bias
        self.arg317_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").q_bias
        self.arg318_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").single_layer_norm.lin_s_merged.weight
        self.arg319_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").to_qkv.weight
        self.arg320_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.weight
        self.arg321_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.bias
        self.arg322_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").q_bias
        self.arg323_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").single_layer_norm.lin_s_merged.weight
        self.arg324_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").to_qkv.weight
        self.arg325_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.weight
        self.arg326_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.bias
        self.arg327_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").q_bias
        self.arg328_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").single_layer_norm.lin_s_merged.weight
        self.arg329_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").to_qkv.weight
        self.arg330_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.weight
        self.arg331_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.bias
        self.arg332_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").weight
        self.arg333_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").bias
        self.arg334_1 = getattr(model.atom_attention_decoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "1").weight
        self.arg335_1 = getattr(model.atom_attention_decoder.to_pos_updates, "0").weight
        self.arg336_1 = getattr(model.atom_attention_decoder.to_pos_updates, "0").bias
        self.arg337_1 = getattr(model.atom_attention_decoder.to_pos_updates, "1").weight
        self.arg338_1 = model.structure_cond_to_token_structure_proj.weight
        self.arg339_1 = model.post_attn_layernorm.weight
        self.arg340_1 = model.post_attn_layernorm.bias
        self.arg341_1 = model.post_atom_cond_layernorm.weight
        self.arg342_1 = model.post_atom_cond_layernorm.bias

    def forward(self, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1):
        arg0_1 = self.arg0_1
        arg1_1 = self.arg1_1
        arg2_1 = self.arg2_1
        arg3_1 = self.arg3_1
        arg4_1 = self.arg4_1
        arg5_1 = self.arg5_1
        arg6_1 = self.arg6_1
        arg7_1 = self.arg7_1
        arg8_1 = self.arg8_1
        arg9_1 = self.arg9_1
        arg10_1 = self.arg10_1
        arg11_1 = self.arg11_1
        arg12_1 = self.arg12_1
        arg13_1 = self.arg13_1
        arg14_1 = self.arg14_1
        arg15_1 = self.arg15_1
        arg16_1 = self.arg16_1
        arg17_1 = self.arg17_1
        arg18_1 = self.arg18_1
        arg19_1 = self.arg19_1
        arg20_1 = self.arg20_1
        arg21_1 = self.arg21_1
        arg22_1 = self.arg22_1
        arg23_1 = self.arg23_1
        arg24_1 = self.arg24_1
        arg25_1 = self.arg25_1
        arg26_1 = self.arg26_1
        arg27_1 = self.arg27_1
        arg28_1 = self.arg28_1
        arg29_1 = self.arg29_1
        arg30_1 = self.arg30_1
        arg31_1 = self.arg31_1
        arg32_1 = self.arg32_1
        arg33_1 = self.arg33_1
        arg34_1 = self.arg34_1
        arg35_1 = self.arg35_1
        arg36_1 = self.arg36_1
        arg37_1 = self.arg37_1
        arg38_1 = self.arg38_1
        arg39_1 = self.arg39_1
        arg40_1 = self.arg40_1
        arg41_1 = self.arg41_1
        arg42_1 = self.arg42_1
        arg43_1 = self.arg43_1
        arg44_1 = self.arg44_1
        arg45_1 = self.arg45_1
        arg46_1 = self.arg46_1
        arg47_1 = self.arg47_1
        arg48_1 = self.arg48_1
        arg49_1 = self.arg49_1
        arg50_1 = self.arg50_1
        arg51_1 = self.arg51_1
        arg52_1 = self.arg52_1
        arg53_1 = self.arg53_1
        arg54_1 = self.arg54_1
        arg55_1 = self.arg55_1
        arg56_1 = self.arg56_1
        arg57_1 = self.arg57_1
        arg58_1 = self.arg58_1
        arg59_1 = self.arg59_1
        arg60_1 = self.arg60_1
        arg61_1 = self.arg61_1
        arg62_1 = self.arg62_1
        arg63_1 = self.arg63_1
        arg64_1 = self.arg64_1
        arg65_1 = self.arg65_1
        arg66_1 = self.arg66_1
        arg67_1 = self.arg67_1
        arg68_1 = self.arg68_1
        arg69_1 = self.arg69_1
        arg70_1 = self.arg70_1
        arg71_1 = self.arg71_1
        arg72_1 = self.arg72_1
        arg73_1 = self.arg73_1
        arg74_1 = self.arg74_1
        arg75_1 = self.arg75_1
        arg76_1 = self.arg76_1
        arg77_1 = self.arg77_1
        arg78_1 = self.arg78_1
        arg79_1 = self.arg79_1
        arg80_1 = self.arg80_1
        arg81_1 = self.arg81_1
        arg82_1 = self.arg82_1
        arg83_1 = self.arg83_1
        arg84_1 = self.arg84_1
        arg85_1 = self.arg85_1
        arg86_1 = self.arg86_1
        arg87_1 = self.arg87_1
        arg88_1 = self.arg88_1
        arg89_1 = self.arg89_1
        arg90_1 = self.arg90_1
        arg91_1 = self.arg91_1
        arg92_1 = self.arg92_1
        arg93_1 = self.arg93_1
        arg94_1 = self.arg94_1
        arg95_1 = self.arg95_1
        arg96_1 = self.arg96_1
        arg97_1 = self.arg97_1
        arg98_1 = self.arg98_1
        arg99_1 = self.arg99_1
        arg100_1 = self.arg100_1
        arg101_1 = self.arg101_1
        arg102_1 = self.arg102_1
        arg103_1 = self.arg103_1
        arg104_1 = self.arg104_1
        arg105_1 = self.arg105_1
        arg106_1 = self.arg106_1
        arg107_1 = self.arg107_1
        arg108_1 = self.arg108_1
        arg109_1 = self.arg109_1
        arg110_1 = self.arg110_1
        arg111_1 = self.arg111_1
        arg112_1 = self.arg112_1
        arg113_1 = self.arg113_1
        arg114_1 = self.arg114_1
        arg115_1 = self.arg115_1
        arg116_1 = self.arg116_1
        arg117_1 = self.arg117_1
        arg118_1 = self.arg118_1
        arg119_1 = self.arg119_1
        arg120_1 = self.arg120_1
        arg121_1 = self.arg121_1
        arg122_1 = self.arg122_1
        arg123_1 = self.arg123_1
        arg124_1 = self.arg124_1
        arg125_1 = self.arg125_1
        arg126_1 = self.arg126_1
        arg127_1 = self.arg127_1
        arg128_1 = self.arg128_1
        arg129_1 = self.arg129_1
        arg130_1 = self.arg130_1
        arg131_1 = self.arg131_1
        arg132_1 = self.arg132_1
        arg133_1 = self.arg133_1
        arg134_1 = self.arg134_1
        arg135_1 = self.arg135_1
        arg136_1 = self.arg136_1
        arg137_1 = self.arg137_1
        arg138_1 = self.arg138_1
        arg139_1 = self.arg139_1
        arg140_1 = self.arg140_1
        arg141_1 = self.arg141_1
        arg142_1 = self.arg142_1
        arg143_1 = self.arg143_1
        arg144_1 = self.arg144_1
        arg145_1 = self.arg145_1
        arg146_1 = self.arg146_1
        arg147_1 = self.arg147_1
        arg148_1 = self.arg148_1
        arg149_1 = self.arg149_1
        arg150_1 = self.arg150_1
        arg151_1 = self.arg151_1
        arg152_1 = self.arg152_1
        arg153_1 = self.arg153_1
        arg154_1 = self.arg154_1
        arg155_1 = self.arg155_1
        arg156_1 = self.arg156_1
        arg157_1 = self.arg157_1
        arg158_1 = self.arg158_1
        arg159_1 = self.arg159_1
        arg160_1 = self.arg160_1
        arg161_1 = self.arg161_1
        arg162_1 = self.arg162_1
        arg163_1 = self.arg163_1
        arg164_1 = self.arg164_1
        arg165_1 = self.arg165_1
        arg166_1 = self.arg166_1
        arg167_1 = self.arg167_1
        arg168_1 = self.arg168_1
        arg169_1 = self.arg169_1
        arg170_1 = self.arg170_1
        arg171_1 = self.arg171_1
        arg172_1 = self.arg172_1
        arg173_1 = self.arg173_1
        arg174_1 = self.arg174_1
        arg175_1 = self.arg175_1
        arg176_1 = self.arg176_1
        arg177_1 = self.arg177_1
        arg178_1 = self.arg178_1
        arg179_1 = self.arg179_1
        arg180_1 = self.arg180_1
        arg181_1 = self.arg181_1
        arg182_1 = self.arg182_1
        arg183_1 = self.arg183_1
        arg184_1 = self.arg184_1
        arg185_1 = self.arg185_1
        arg186_1 = self.arg186_1
        arg187_1 = self.arg187_1
        arg188_1 = self.arg188_1
        arg189_1 = self.arg189_1
        arg190_1 = self.arg190_1
        arg191_1 = self.arg191_1
        arg192_1 = self.arg192_1
        arg193_1 = self.arg193_1
        arg194_1 = self.arg194_1
        arg195_1 = self.arg195_1
        arg196_1 = self.arg196_1
        arg197_1 = self.arg197_1
        arg198_1 = self.arg198_1
        arg199_1 = self.arg199_1
        arg200_1 = self.arg200_1
        arg201_1 = self.arg201_1
        arg202_1 = self.arg202_1
        arg203_1 = self.arg203_1
        arg204_1 = self.arg204_1
        arg205_1 = self.arg205_1
        arg206_1 = self.arg206_1
        arg207_1 = self.arg207_1
        arg208_1 = self.arg208_1
        arg209_1 = self.arg209_1
        arg210_1 = self.arg210_1
        arg211_1 = self.arg211_1
        arg212_1 = self.arg212_1
        arg213_1 = self.arg213_1
        arg214_1 = self.arg214_1
        arg215_1 = self.arg215_1
        arg216_1 = self.arg216_1
        arg217_1 = self.arg217_1
        arg218_1 = self.arg218_1
        arg219_1 = self.arg219_1
        arg220_1 = self.arg220_1
        arg221_1 = self.arg221_1
        arg222_1 = self.arg222_1
        arg223_1 = self.arg223_1
        arg224_1 = self.arg224_1
        arg225_1 = self.arg225_1
        arg226_1 = self.arg226_1
        arg227_1 = self.arg227_1
        arg228_1 = self.arg228_1
        arg229_1 = self.arg229_1
        arg230_1 = self.arg230_1
        arg231_1 = self.arg231_1
        arg232_1 = self.arg232_1
        arg233_1 = self.arg233_1
        arg234_1 = self.arg234_1
        arg235_1 = self.arg235_1
        arg236_1 = self.arg236_1
        arg237_1 = self.arg237_1
        arg238_1 = self.arg238_1
        arg239_1 = self.arg239_1
        arg240_1 = self.arg240_1
        arg241_1 = self.arg241_1
        arg242_1 = self.arg242_1
        arg243_1 = self.arg243_1
        arg244_1 = self.arg244_1
        arg245_1 = self.arg245_1
        arg246_1 = self.arg246_1
        arg247_1 = self.arg247_1
        arg248_1 = self.arg248_1
        arg249_1 = self.arg249_1
        arg250_1 = self.arg250_1
        arg251_1 = self.arg251_1
        arg252_1 = self.arg252_1
        arg253_1 = self.arg253_1
        arg254_1 = self.arg254_1
        arg255_1 = self.arg255_1
        arg256_1 = self.arg256_1
        arg257_1 = self.arg257_1
        arg258_1 = self.arg258_1
        arg259_1 = self.arg259_1
        arg260_1 = self.arg260_1
        arg261_1 = self.arg261_1
        arg262_1 = self.arg262_1
        arg263_1 = self.arg263_1
        arg264_1 = self.arg264_1
        arg265_1 = self.arg265_1
        arg266_1 = self.arg266_1
        arg267_1 = self.arg267_1
        arg268_1 = self.arg268_1
        arg269_1 = self.arg269_1
        arg270_1 = self.arg270_1
        arg271_1 = self.arg271_1
        arg272_1 = self.arg272_1
        arg273_1 = self.arg273_1
        arg274_1 = self.arg274_1
        arg275_1 = self.arg275_1
        arg276_1 = self.arg276_1
        arg277_1 = self.arg277_1
        arg278_1 = self.arg278_1
        arg279_1 = self.arg279_1
        arg280_1 = self.arg280_1
        arg281_1 = self.arg281_1
        arg282_1 = self.arg282_1
        arg283_1 = self.arg283_1
        arg284_1 = self.arg284_1
        arg285_1 = self.arg285_1
        arg286_1 = self.arg286_1
        arg287_1 = self.arg287_1
        arg288_1 = self.arg288_1
        arg289_1 = self.arg289_1
        arg290_1 = self.arg290_1
        arg291_1 = self.arg291_1
        arg292_1 = self.arg292_1
        arg293_1 = self.arg293_1
        arg294_1 = self.arg294_1
        arg295_1 = self.arg295_1
        arg296_1 = self.arg296_1
        arg297_1 = self.arg297_1
        arg298_1 = self.arg298_1
        arg299_1 = self.arg299_1
        arg300_1 = self.arg300_1
        arg301_1 = self.arg301_1
        arg302_1 = self.arg302_1
        arg303_1 = self.arg303_1
        arg304_1 = self.arg304_1
        arg305_1 = self.arg305_1
        arg306_1 = self.arg306_1
        arg307_1 = self.arg307_1
        arg308_1 = self.arg308_1
        arg309_1 = self.arg309_1
        arg310_1 = self.arg310_1
        arg311_1 = self.arg311_1
        arg312_1 = self.arg312_1
        arg313_1 = self.arg313_1
        arg314_1 = self.arg314_1
        arg315_1 = self.arg315_1
        arg316_1 = self.arg316_1
        arg317_1 = self.arg317_1
        arg318_1 = self.arg318_1
        arg319_1 = self.arg319_1
        arg320_1 = self.arg320_1
        arg321_1 = self.arg321_1
        arg322_1 = self.arg322_1
        arg323_1 = self.arg323_1
        arg324_1 = self.arg324_1
        arg325_1 = self.arg325_1
        arg326_1 = self.arg326_1
        arg327_1 = self.arg327_1
        arg328_1 = self.arg328_1
        arg329_1 = self.arg329_1
        arg330_1 = self.arg330_1
        arg331_1 = self.arg331_1
        arg332_1 = self.arg332_1
        arg333_1 = self.arg333_1
        arg334_1 = self.arg334_1
        arg335_1 = self.arg335_1
        arg336_1 = self.arg336_1
        arg337_1 = self.arg337_1
        arg338_1 = self.arg338_1
        arg339_1 = self.arg339_1
        arg340_1 = self.arg340_1
        arg341_1 = self.arg341_1
        arg342_1 = self.arg342_1

        cat = torch.cat([arg346_1,arg344_1],dim = -1) ;  arg346_1 = arg344_1 = None
        native_layer_norm_default = (torch.nn.functional.layer_norm(cat,[512],arg0_1,arg1_1,1e-05),) ;  cat = arg0_1 = arg1_1 = None
        getitem = native_layer_norm_default[0]
        t = arg2_1.t() ;  arg2_1 = None
        view = getitem.view(262144, 512) ;  getitem = None
        mm = torch.mm(view,t) ;  view = t = None
        view_1 = mm.view(1, 512, 512, 256) ;  mm = None
        split_tensor = torch.split(view_1,512,dim = -2) 
        getitem_3 = split_tensor[0];  split_tensor = None
        native_layer_norm_default_1 = (torch.nn.functional.layer_norm(getitem_3,[256],arg10_1,arg11_1,1e-05),) ;  getitem_3 = arg10_1 = arg11_1 = None
        getitem_4 = native_layer_norm_default_1[0]
        t_1 = arg12_1.t() ;  arg12_1 = None
        view_2 = getitem_4.view(262144, 256) ;  getitem_4 = None
        mm_1 = torch.mm(view_2,t_1) ;  view_2 = t_1 = None
        view_3 = mm_1.view(1, 512, 512, 1024) ;  mm_1 = None
        split_tensor_1 = torch.split(view_3,512,dim = -1) ;  view_3 = None
        getitem_7 = split_tensor_1[0]
        getitem_8 = split_tensor_1[1];  split_tensor_1 = None
        silu = torch.nn.functional.silu(getitem_7) ;  getitem_7 = None
        mul = torch.mul(silu,getitem_8) ;  silu = getitem_8 = None
        t_2 = arg13_1.t() ;  arg13_1 = None
        view_5 = mul.view(262144, 512) ;  mul = None
        mm_2 = torch.mm(view_5,t_2) ;  view_5 = t_2 = None
        view_6 = mm_2.view(1, 512, 512, 256) ;  mm_2 = None
        add = torch.add(view_1,view_6) ;  view_1 = view_6 = None
        split_tensor_2 = torch.split(add,512,dim = -2) 
        getitem_9 = split_tensor_2[0];  split_tensor_2 = None
        native_layer_norm_default_2 = (torch.nn.functional.layer_norm(getitem_9,[256],arg18_1,arg19_1,1e-05),) ;  getitem_9 = arg18_1 = arg19_1 = None
        getitem_10 = native_layer_norm_default_2[0]
        t_3 = arg20_1.t() ;  arg20_1 = None
        view_7 = getitem_10.view(262144, 256) ;  getitem_10 = None
        mm_3 = torch.mm(view_7,t_3) ;  view_7 = t_3 = None
        view_8 = mm_3.view(1, 512, 512, 1024) ;  mm_3 = None
        split_tensor_3 = torch.split(view_8,512,dim = -1) ;  view_8 = None
        getitem_13 = split_tensor_3[0]
        getitem_14 = split_tensor_3[1];  split_tensor_3 = None
        silu_1 = torch.nn.functional.silu(getitem_13) ;  getitem_13 = None
        mul_1 = torch.mul(silu_1,getitem_14) ;  silu_1 = getitem_14 = None
        t_4 = arg21_1.t() ;  arg21_1 = None
        view_10 = mul_1.view(262144, 512) ;  mul_1 = None
        mm_4 = torch.mm(view_10,t_4) ;  view_10 = t_4 = None
        view_11 = mm_4.view(1, 512, 512, 256) ;  mm_4 = None
        add_1 = torch.add(add,view_11) ;  add = view_11 = None
        cat_1 = torch.cat([arg343_1,arg345_1],dim = -1) ;  arg343_1 = None
        native_layer_norm_default_3 = (torch.nn.functional.layer_norm(cat_1,[768],arg3_1,arg4_1,1e-05),) ;  cat_1 = arg3_1 = arg4_1 = None
        getitem_15 = native_layer_norm_default_3[0]
        t_5 = arg5_1.t() ;  arg5_1 = None
        view_12 = getitem_15.view(512, 768) ;  getitem_15 = None
        mm_5 = torch.mm(view_12,t_5) ;  view_12 = t_5 = None
        view_13 = mm_5.view(1, 512, 384) ;  mm_5 = None
        clamp_min = torch.clamp_min(arg355_1,1.1920928955078125e-07) 
        log = torch.log(clamp_min) ;  clamp_min = None
        mul_2 = torch.mul(log,0.25) ;  log = None
        view_14 = mul_2.view(5,) ;  mul_2 = None
        view_15 = view_14.view(5, 1) ;  view_14 = None
        mul_3 = torch.mul(view_15,arg22_1) ;  view_15 = arg22_1 = None
        add_2 = torch.add(mul_3,arg23_1) ;  mul_3 = arg23_1 = None
        mul_4 = torch.mul(add_2,6.283185307179586) ;  add_2 = None
        cos = torch.cos(mul_4) ;  mul_4 = None
        view_16 = cos.view(5, 1, 256) ;  cos = None
        native_layer_norm_default_4 = (torch.nn.functional.layer_norm(view_16,[256],arg24_1,arg25_1,1e-05),) ;  view_16 = arg24_1 = arg25_1 = None
        getitem_18 = native_layer_norm_default_4[0]
        t_6 = arg26_1.t() ;  arg26_1 = None
        view_17 = getitem_18.view(5, 256) ;  getitem_18 = None
        mm_6 = torch.mm(view_17,t_6) ;  view_17 = t_6 = None
        view_18 = mm_6.view(5, 1, 384) ;  mm_6 = None
        view_19 = view_13.view(1, 1, 512, 384) ;  view_13 = None
        view_20 = view_18.view(1, 5, 1, 384) ;  view_18 = None
        add_3 = torch.add(view_19,view_20) ;  view_19 = view_20 = None
        split_tensor_4 = torch.split(add_3,512,dim = -2) 
        getitem_21 = split_tensor_4[0];  split_tensor_4 = None
        native_layer_norm_default_5 = (torch.nn.functional.layer_norm(getitem_21,[384],arg6_1,arg7_1,1e-05),) ;  getitem_21 = arg6_1 = arg7_1 = None
        getitem_22 = native_layer_norm_default_5[0]
        t_7 = arg8_1.t() ;  arg8_1 = None
        view_21 = getitem_22.view(2560, 384) ;  getitem_22 = None
        mm_7 = torch.mm(view_21,t_7) ;  view_21 = t_7 = None
        view_22 = mm_7.view(1, 5, 512, 1536) ;  mm_7 = None
        split_tensor_5 = torch.split(view_22,768,dim = -1) ;  view_22 = None
        getitem_25 = split_tensor_5[0]
        getitem_26 = split_tensor_5[1];  split_tensor_5 = None
        silu_2 = torch.nn.functional.silu(getitem_25) ;  getitem_25 = None
        mul_5 = torch.mul(silu_2,getitem_26) ;  silu_2 = getitem_26 = None
        t_8 = arg9_1.t() ;  arg9_1 = None
        view_24 = mul_5.view(2560, 768) ;  mul_5 = None
        mm_8 = torch.mm(view_24,t_8) ;  view_24 = t_8 = None
        view_25 = mm_8.view(1, 5, 512, 384) ;  mm_8 = None
        add_4 = torch.add(add_3,view_25) ;  add_3 = view_25 = None
        split_tensor_6 = torch.split(add_4,512,dim = -2) 
        getitem_27 = split_tensor_6[0];  split_tensor_6 = None
        native_layer_norm_default_6 = (torch.nn.functional.layer_norm(getitem_27,[384],arg14_1,arg15_1,1e-05),) ;  getitem_27 = arg14_1 = arg15_1 = None
        getitem_28 = native_layer_norm_default_6[0]
        t_9 = arg16_1.t() ;  arg16_1 = None
        view_26 = getitem_28.view(2560, 384) ;  getitem_28 = None
        mm_9 = torch.mm(view_26,t_9) ;  view_26 = t_9 = None
        view_27 = mm_9.view(1, 5, 512, 1536) ;  mm_9 = None
        split_tensor_7 = torch.split(view_27,768,dim = -1) ;  view_27 = None
        getitem_31 = split_tensor_7[0]
        getitem_32 = split_tensor_7[1];  split_tensor_7 = None
        silu_3 = torch.nn.functional.silu(getitem_31) ;  getitem_31 = None
        mul_6 = torch.mul(silu_3,getitem_32) ;  silu_3 = getitem_32 = None
        t_10 = arg17_1.t() ;  arg17_1 = None
        view_29 = mul_6.view(2560, 768) ;  mul_6 = None
        mm_10 = torch.mm(view_29,t_10) ;  view_29 = t_10 = None
        view_30 = mm_10.view(1, 5, 512, 384) ;  mm_10 = None
        add_5 = torch.add(add_4,view_30) ;  add_4 = view_30 = None
        native_layer_norm_default_7 = (torch.nn.functional.layer_norm(add_5,[384],arg27_1,arg28_1,1e-05),) ;  add_5 = arg27_1 = arg28_1 = None
        getitem_33 = native_layer_norm_default_7[0]
        native_layer_norm_default_8 = (torch.nn.functional.layer_norm(add_1,[256],arg29_1,arg30_1,1e-05),) ;  add_1 = arg29_1 = arg30_1 = None
        getitem_36 = native_layer_norm_default_8[0]
        view_31 = arg355_1.view(1, 5, 1, 1) ;  arg355_1 = None
        pow_1 = torch.pow(view_31,2) 
        add_6 = torch.add(pow_1,256.0) ;  pow_1 = None
        pow_2 = torch.pow(add_6,-0.5) ;  add_6 = None
        mul_7 = torch.mul(arg354_1,pow_2) ;  pow_2 = None
        t_11 = arg31_1.t() ;  arg31_1 = None
        view_32 = arg347_1.view(11776, 128) ;  arg347_1 = None
        mm_11 = torch.mm(view_32,t_11) ;  view_32 = t_11 = None
        view_33 = mm_11.view(1, 11776, 128) ;  mm_11 = None
        unsqueeze = torch.unsqueeze(view_33,1) 
        expand = unsqueeze.expand(-1, 1, -1, -1) ;  unsqueeze = None
        native_layer_norm_default_9 = (torch.nn.functional.layer_norm(arg345_1,[384],arg32_1,arg33_1,1e-05),) ;  arg345_1 = arg32_1 = arg33_1 = None
        getitem_39 = native_layer_norm_default_9[0]
        t_12 = arg34_1.t() ;  arg34_1 = None
        view_34 = getitem_39.view(512, 384) ;  getitem_39 = None
        mm_12 = torch.mm(view_34,t_12) ;  view_34 = t_12 = None
        view_35 = mm_12.view(1, 512, 128) ;  mm_12 = None
        arange = torch.arange(1,device = self.device,pin_memory = False) 
        unsqueeze_1 = torch.unsqueeze(arange,1) ;  arange = None
        index = view_35[unsqueeze_1,arg356_1] ;  view_35 = unsqueeze_1 = None
        clone = torch.clone(index) ;  index = None
        add_7 = torch.add(view_33,clone) ;  view_33 = clone = None
        native_layer_norm_default_10 = (torch.nn.functional.layer_norm(add_7,[128],None,None,1e-05),) ;  add_7 = None
        getitem_42 = native_layer_norm_default_10[0]
        t_13 = arg35_1.t() ;  arg35_1 = None
        view_36 = mul_7.view(58880, 3) ;  mul_7 = None
        mm_13 = torch.mm(view_36,t_13) ;  view_36 = t_13 = None
        view_37 = mm_13.view(1, 5, 11776, 128) ;  mm_13 = None
        add_8 = torch.add(expand,view_37) ;  expand = view_37 = None
        native_layer_norm_default_11 = (torch.nn.functional.layer_norm(getitem_36,[256],arg74_1,arg75_1,1e-05),) ;  arg74_1 = arg75_1 = None
        getitem_45 = native_layer_norm_default_11[0]
        t_14 = arg76_1.t() ;  arg76_1 = None
        view_38 = getitem_45.view(262144, 256) ;  getitem_45 = None
        mm_14 = torch.mm(view_38,t_14) ;  view_38 = t_14 = None
        view_39 = mm_14.view(1, 512, 512, 16) ;  mm_14 = None
        unsqueeze_2 = torch.unsqueeze(getitem_42,1) 
        expand_1 = unsqueeze_2.expand(-1, 5, -1, -1) ;  unsqueeze_2 = None
        slice_1 = expand_1[0:] 
        slice_2 = slice_1[:, 0:] ;  slice_1 = None
        index_1 = slice_2[:,:,arg352_1] ;  slice_2 = None
        slice_3 = expand_1[0:] ;  expand_1 = None
        slice_4 = slice_3[:, 0:] ;  slice_3 = None
        index_2 = slice_4[:,:,arg353_1] ;  slice_4 = None
        slice_5 = arg356_1[0:] 
        index_3 = slice_5[:,arg352_1] ;  slice_5 = arg352_1 = None
        slice_6 = arg356_1[0:] 
        index_4 = slice_6[:,arg353_1] ;  slice_6 = arg353_1 = None
        view_40 = index_3.view(1, 368, 32, 1) ;  index_3 = None
        view_41 = index_4.view(1, 368, 1, 128) ;  index_4 = None
        arange_1 = torch.arange(1,device = self.device,pin_memory = False) 
        view_42 = arange_1.view(-1, 1, 1, 1) ;  arange_1 = None
        index_5 = view_39[view_42,view_40,view_41] ;  view_39 = view_42 = view_40 = view_41 = None
        add_9 = torch.add(arg348_1,index_5) ;  arg348_1 = index_5 = None
        relu = torch.relu(index_1) ;  index_1 = None
        t_15 = arg36_1.t() ;  arg36_1 = None
        view_43 = relu.view(58880, 128) ;  relu = None
        mm_15 = torch.mm(view_43,t_15) ;  view_43 = t_15 = None
        view_44 = mm_15.view(1, 5, 368, 32, 16) ;  mm_15 = None
        relu_1 = torch.relu(index_2) ;  index_2 = None
        t_16 = arg37_1.t() ;  arg37_1 = None
        view_45 = relu_1.view(235520, 128) ;  relu_1 = None
        mm_16 = torch.mm(view_45,t_16) ;  view_45 = t_16 = None
        view_46 = mm_16.view(1, 5, 368, 128, 16) ;  mm_16 = None
        unsqueeze_3 = torch.unsqueeze(add_9,1) ;  add_9 = None
        view_47 = view_44.view(1, 5, 368, 32, 1, 16) ;  view_44 = None
        add_10 = torch.add(unsqueeze_3,view_47) ;  unsqueeze_3 = view_47 = None
        view_48 = view_46.view(1, 5, 368, 1, 128, 16) ;  view_46 = None
        add_11 = torch.add(add_10,view_48) ;  add_10 = view_48 = None
        t_17 = arg38_1.t() ;  arg38_1 = None
        view_49 = add_11.view(7536640, 16) 
        mm_17 = torch.mm(view_49,t_17) ;  view_49 = t_17 = None
        view_50 = mm_17.view(1, 5, 368, 32, 128, 16) ;  mm_17 = None
        relu_2 = torch.relu(view_50) ;  view_50 = None
        view_51 = relu_2.view(7536640, 16) ;  relu_2 = None
        t_18 = arg39_1.t() ;  arg39_1 = None
        view_54 = view_51.view(1, 5, 368, 32, 128, 16) ;  view_51 = None
        view_55 = view_54.view(7536640, 16) ;  view_54 = None
        mm_18 = torch.mm(view_55,t_18) ;  view_55 = t_18 = None
        view_56 = mm_18.view(1, 5, 368, 32, 128, 16) ;  mm_18 = None
        add_12 = torch.add(view_56,add_11) ;  view_56 = add_11 = None
        view_57 = add_12.view(7536640, 16) ;  add_12 = None
        view_59 = add_8.view(5, 11776, 128) ;  add_8 = None
        unsqueeze_4 = torch.unsqueeze(getitem_42,1) 
        expand_2 = unsqueeze_4.expand(-1, 5, -1, -1) ;  unsqueeze_4 = None
        view_60 = expand_2.view(5, 11776, 128) ;  expand_2 = None
        unsqueeze_5 = torch.unsqueeze(arg350_1,1) 
        expand_3 = unsqueeze_5.expand(-1, 5, -1, -1, -1) ;  unsqueeze_5 = None
        view_62 = expand_3.view(5, 368, 32, 128) ;  expand_3 = None
        unsqueeze_6 = torch.unsqueeze(arg349_1,1) 
        expand_4 = unsqueeze_6.expand(-1, 5, -1) ;  unsqueeze_6 = None
        view_63 = expand_4.view(5, 11776) ;  expand_4 = None
        arange_2 = torch.arange(11776,device = self.device,pin_memory = False) 
        view_64 = arange_2.view(368, 32) ;  arange_2 = None
        slice_7 = view_64[0:] ;  view_64 = None
        slice_8 = slice_7[:, 0:1] ;  slice_7 = None
        add_13 = torch.add(slice_8,-48) ;  slice_8 = None
        arange_3 = torch.arange(128,device = self.device,pin_memory = False) 
        add_14 = torch.add(add_13,arange_3) ;  add_13 = arange_3 = None
        remainder = torch.remainder(add_14,11776) ;  add_14 = None
        view_65 = view_57.view(1, 5, 368, 32, 128, 16) 
        view_66 = view_65.view(5, 368, 32, 128, 16) ;  view_65 = None
        native_layer_norm_default_12 = (torch.nn.functional.layer_norm(view_66,[16],arg70_1,arg71_1,1e-05),) ;  view_66 = arg70_1 = arg71_1 = None
        getitem_48 = native_layer_norm_default_12[0]
        unbind_int = torch.unbind(arg72_1) ;  arg72_1 = None
        getitem_51 = unbind_int[0]
        getitem_52 = unbind_int[1]
        getitem_53 = unbind_int[2];  unbind_int = None
        unsqueeze_7 = torch.unsqueeze(view_63,-1) 
        bitwise_not = torch.bitwise_not(unsqueeze_7) ;  unsqueeze_7 = None
        masked_fill = view_59.masked_fill(bitwise_not,0.0) ;  view_59 = bitwise_not = None
        unsqueeze_8 = torch.unsqueeze(getitem_48,5) 
        permute = unsqueeze_8.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_8 = None
        unsqueeze_9 = torch.unsqueeze(getitem_51,2) ;  getitem_51 = None
        unsqueeze_10 = torch.unsqueeze(unsqueeze_9,3) ;  unsqueeze_9 = None
        unsqueeze_11 = torch.unsqueeze(unsqueeze_10,4) ;  unsqueeze_10 = None
        unsqueeze_12 = torch.unsqueeze(unsqueeze_11,5) ;  unsqueeze_11 = None
        permute_1 = unsqueeze_12.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_12 = None
        permute_2 = permute.permute(0, 2, 3, 4, 5, 1) ;  permute = None
        view_67 = permute_2.view(1, 7536640, 16) ;  permute_2 = None
        permute_3 = permute_1.permute(5, 1, 0, 2, 3, 4) ;  permute_1 = None
        view_68 = permute_3.view(1, 16, 4) ;  permute_3 = None
        bmm = torch.bmm(view_67,view_68) ;  view_67 = view_68 = None
        view_69 = bmm.view(5, 368, 32, 128, 1, 4) ;  bmm = None
        permute_4 = view_69.permute(0, 5, 1, 2, 3, 4) ;  view_69 = None
        view_70 = permute_4.view(5, 4, 368, 32, 128) ;  permute_4 = None
        view_71 = view_62.view(5, 1, 368, 32, 128) 
        bitwise_not_1 = torch.bitwise_not(view_71) ;  view_71 = None
        masked_fill_1 = view_70.masked_fill(bitwise_not_1,-10000) ;  view_70 = bitwise_not_1 = None
        native_layer_norm_default_13 = (torch.nn.functional.layer_norm(masked_fill,[128],None,None,0.1),) 
        getitem_54 = native_layer_norm_default_13[0]
        t_19 = arg56_1.t() ;  arg56_1 = None
        clone_1 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view = clone_1.view(58880, 128) ;  clone_1 = None
        mm_19 = torch.mm(_unsafe_view,t_19) ;  _unsafe_view = t_19 = None
        view_72 = mm_19.view(5, 11776, 256) ;  mm_19 = None
        split_tensor_8 = torch.split(view_72,128,dim = -1) ;  view_72 = None
        getitem_57 = split_tensor_8[0]
        getitem_58 = split_tensor_8[1];  split_tensor_8 = None
        add_15 = torch.add(getitem_57,1) ;  getitem_57 = None
        mul_8 = torch.mul(getitem_54,add_15) ;  getitem_54 = add_15 = None
        add_16 = torch.add(mul_8,getitem_58) ;  mul_8 = getitem_58 = None
        unsqueeze_13 = torch.unsqueeze(add_16,3) ;  add_16 = None
        unsqueeze_14 = torch.unsqueeze(unsqueeze_13,4) ;  unsqueeze_13 = None
        unsqueeze_15 = torch.unsqueeze(unsqueeze_14,5) ;  unsqueeze_14 = None
        permute_5 = unsqueeze_15.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_15 = None
        unsqueeze_16 = torch.unsqueeze(arg57_1,4) ;  arg57_1 = None
        unsqueeze_17 = torch.unsqueeze(unsqueeze_16,5) ;  unsqueeze_16 = None
        permute_6 = unsqueeze_17.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_17 = None
        permute_7 = permute_5.permute(1, 3, 5, 0, 2, 4) ;  permute_5 = None
        view_73 = permute_7.view(1, 58880, 128) ;  permute_7 = None
        permute_8 = permute_6.permute(5, 0, 2, 4, 1, 3) ;  permute_6 = None
        view_74 = permute_8.view(1, 128, 384) ;  permute_8 = None
        bmm_1 = torch.bmm(view_73,view_74) ;  view_73 = view_74 = None
        view_75 = bmm_1.view(5, 11776, 1, 3, 4, 32) ;  bmm_1 = None
        permute_9 = view_75.permute(3, 0, 4, 1, 5, 2) ;  view_75 = None
        view_76 = permute_9.view(3, 5, 4, 11776, 32) ;  permute_9 = None
        clone_2 = torch.clone(view_76,memory_format = torch.contiguous_format) ;  view_76 = None
        _unsafe_view_1 = clone_2.view(3, 20, 11776, 32) ;  clone_2 = None
        unbind_int_1 = torch.unbind(_unsafe_view_1) ;  _unsafe_view_1 = None
        getitem_59 = unbind_int_1[0]
        getitem_60 = unbind_int_1[1]
        getitem_61 = unbind_int_1[2];  unbind_int_1 = None
        unsqueeze_18 = torch.unsqueeze(arg55_1,0) ;  arg55_1 = None
        expand_5 = unsqueeze_18.expand(5, -1, -1) ;  unsqueeze_18 = None
        clone_3 = torch.clone(expand_5,memory_format = torch.contiguous_format) ;  expand_5 = None
        _unsafe_view_2 = clone_3.view(20, 1, 32) ;  clone_3 = None
        add_17 = torch.add(getitem_59,_unsafe_view_2) ;  getitem_59 = _unsafe_view_2 = None
        view_77 = add_17.view(20, 368, 32, 32) ;  add_17 = None
        slice_9 = getitem_60[0:] ;  getitem_60 = None
        slice_10 = slice_9[:, :, 0:] ;  slice_9 = None
        index_6 = slice_10[:,remainder] ;  slice_10 = None
        slice_11 = getitem_61[0:] ;  getitem_61 = None
        slice_12 = slice_11[:, :, 0:] ;  slice_11 = None
        index_7 = slice_12[:,remainder] ;  slice_12 = None
        view_78 = masked_fill_1.view(20, 368, 32, 128) ;  masked_fill_1 = None
        expand_6 = view_78.expand(20, 368, 32, 128) ;  view_78 = None
        _scaled_dot_product_efficient_attention_default = (torch.nn.functional.scaled_dot_product_attention(view_77,index_6,index_7,expand_6,False),) ;  view_77 = index_6 = index_7 = expand_6 = None
        getitem_62 = _scaled_dot_product_efficient_attention_default[0]
        view_79 = getitem_62.view(5, 4, 368, 32, 32) ;  getitem_62 = None
        permute_10 = view_79.permute(0, 2, 3, 1, 4) ;  view_79 = None
        clone_4 = torch.clone(permute_10,memory_format = torch.contiguous_format) ;  permute_10 = None
        _unsafe_view_3 = clone_4.view(5, 11776, 128) ;  clone_4 = None
        t_20 = arg58_1.t() ;  arg58_1 = None
        clone_5 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_4 = clone_5.view(58880, 128) ;  clone_5 = None
        mm_20 = torch.mm(_unsafe_view_4,t_20) ;  _unsafe_view_4 = t_20 = None
        view_80 = mm_20.view(5, 11776, 128) ;  mm_20 = None
        add_18 = torch.add(view_80,arg59_1) ;  view_80 = arg59_1 = None
        sigmoid = torch.sigmoid(add_18) ;  add_18 = None
        mul_9 = torch.mul(_unsafe_view_3,sigmoid) ;  _unsafe_view_3 = sigmoid = None
        native_layer_norm_default_14 = (torch.nn.functional.layer_norm(masked_fill,[128],None,None,0.1),) 
        getitem_66 = native_layer_norm_default_14[0]
        t_21 = arg40_1.t() ;  arg40_1 = None
        clone_6 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_5 = clone_6.view(58880, 128) ;  clone_6 = None
        mm_21 = torch.mm(_unsafe_view_5,t_21) ;  _unsafe_view_5 = t_21 = None
        view_81 = mm_21.view(5, 11776, 256) ;  mm_21 = None
        split_tensor_9 = torch.split(view_81,128,dim = -1) ;  view_81 = None
        getitem_69 = split_tensor_9[0]
        getitem_70 = split_tensor_9[1];  split_tensor_9 = None
        add_19 = torch.add(getitem_69,1) ;  getitem_69 = None
        mul_10 = torch.mul(getitem_66,add_19) ;  getitem_66 = add_19 = None
        add_20 = torch.add(mul_10,getitem_70) ;  mul_10 = getitem_70 = None
        t_22 = arg41_1.t() ;  arg41_1 = None
        view_82 = add_20.view(58880, 128) ;  add_20 = None
        mm_22 = torch.mm(view_82,t_22) ;  view_82 = t_22 = None
        view_83 = mm_22.view(5, 11776, 512) ;  mm_22 = None
        split_tensor_10 = torch.split(view_83,256,dim = -1) ;  view_83 = None
        getitem_71 = split_tensor_10[0]
        getitem_72 = split_tensor_10[1];  split_tensor_10 = None
        silu_4 = torch.nn.functional.silu(getitem_71) ;  getitem_71 = None
        mul_11 = torch.mul(silu_4,getitem_72) ;  silu_4 = getitem_72 = None
        t_23 = arg43_1.t() ;  arg43_1 = None
        clone_7 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_6 = clone_7.view(58880, 128) ;  clone_7 = None
        mm_23 = torch.mm(_unsafe_view_6,t_23) ;  _unsafe_view_6 = t_23 = None
        view_84 = mm_23.view(5, 11776, 128) ;  mm_23 = None
        add_21 = torch.add(view_84,arg44_1) ;  view_84 = arg44_1 = None
        sigmoid_1 = torch.sigmoid(add_21) ;  add_21 = None
        t_24 = arg42_1.t() ;  arg42_1 = None
        view_85 = mul_11.view(58880, 256) ;  mul_11 = None
        mm_24 = torch.mm(view_85,t_24) ;  view_85 = t_24 = None
        view_86 = mm_24.view(5, 11776, 128) ;  mm_24 = None
        mul_12 = torch.mul(sigmoid_1,view_86) ;  sigmoid_1 = view_86 = None
        add_22 = torch.add(masked_fill,mul_12) ;  masked_fill = mul_12 = None
        add_23 = torch.add(add_22,mul_9) ;  add_22 = mul_9 = None
        unsqueeze_19 = torch.unsqueeze(view_63,-1) 
        bitwise_not_2 = torch.bitwise_not(unsqueeze_19) ;  unsqueeze_19 = None
        masked_fill_2 = add_23.masked_fill(bitwise_not_2,0.0) ;  add_23 = bitwise_not_2 = None
        unsqueeze_20 = torch.unsqueeze(getitem_48,5) 
        permute_11 = unsqueeze_20.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_20 = None
        unsqueeze_21 = torch.unsqueeze(getitem_52,2) ;  getitem_52 = None
        unsqueeze_22 = torch.unsqueeze(unsqueeze_21,3) ;  unsqueeze_21 = None
        unsqueeze_23 = torch.unsqueeze(unsqueeze_22,4) ;  unsqueeze_22 = None
        unsqueeze_24 = torch.unsqueeze(unsqueeze_23,5) ;  unsqueeze_23 = None
        permute_12 = unsqueeze_24.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_24 = None
        permute_13 = permute_11.permute(0, 2, 3, 4, 5, 1) ;  permute_11 = None
        view_87 = permute_13.view(1, 7536640, 16) ;  permute_13 = None
        permute_14 = permute_12.permute(5, 1, 0, 2, 3, 4) ;  permute_12 = None
        view_88 = permute_14.view(1, 16, 4) ;  permute_14 = None
        bmm_2 = torch.bmm(view_87,view_88) ;  view_87 = view_88 = None
        view_89 = bmm_2.view(5, 368, 32, 128, 1, 4) ;  bmm_2 = None
        permute_15 = view_89.permute(0, 5, 1, 2, 3, 4) ;  view_89 = None
        view_90 = permute_15.view(5, 4, 368, 32, 128) ;  permute_15 = None
        view_91 = view_62.view(5, 1, 368, 32, 128) 
        bitwise_not_3 = torch.bitwise_not(view_91) ;  view_91 = None
        masked_fill_3 = view_90.masked_fill(bitwise_not_3,-10000) ;  view_90 = bitwise_not_3 = None
        native_layer_norm_default_15 = (torch.nn.functional.layer_norm(masked_fill_2,[128],None,None,0.1),) 
        getitem_73 = native_layer_norm_default_15[0]
        t_25 = arg61_1.t() ;  arg61_1 = None
        clone_8 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_7 = clone_8.view(58880, 128) ;  clone_8 = None
        mm_25 = torch.mm(_unsafe_view_7,t_25) ;  _unsafe_view_7 = t_25 = None
        view_92 = mm_25.view(5, 11776, 256) ;  mm_25 = None
        split_tensor_11 = torch.split(view_92,128,dim = -1) ;  view_92 = None
        getitem_76 = split_tensor_11[0]
        getitem_77 = split_tensor_11[1];  split_tensor_11 = None
        add_24 = torch.add(getitem_76,1) ;  getitem_76 = None
        mul_13 = torch.mul(getitem_73,add_24) ;  getitem_73 = add_24 = None
        add_25 = torch.add(mul_13,getitem_77) ;  mul_13 = getitem_77 = None
        unsqueeze_25 = torch.unsqueeze(add_25,3) ;  add_25 = None
        unsqueeze_26 = torch.unsqueeze(unsqueeze_25,4) ;  unsqueeze_25 = None
        unsqueeze_27 = torch.unsqueeze(unsqueeze_26,5) ;  unsqueeze_26 = None
        permute_16 = unsqueeze_27.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_27 = None
        unsqueeze_28 = torch.unsqueeze(arg62_1,4) ;  arg62_1 = None
        unsqueeze_29 = torch.unsqueeze(unsqueeze_28,5) ;  unsqueeze_28 = None
        permute_17 = unsqueeze_29.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_29 = None
        permute_18 = permute_16.permute(1, 3, 5, 0, 2, 4) ;  permute_16 = None
        view_93 = permute_18.view(1, 58880, 128) ;  permute_18 = None
        permute_19 = permute_17.permute(5, 0, 2, 4, 1, 3) ;  permute_17 = None
        view_94 = permute_19.view(1, 128, 384) ;  permute_19 = None
        bmm_3 = torch.bmm(view_93,view_94) ;  view_93 = view_94 = None
        view_95 = bmm_3.view(5, 11776, 1, 3, 4, 32) ;  bmm_3 = None
        permute_20 = view_95.permute(3, 0, 4, 1, 5, 2) ;  view_95 = None
        view_96 = permute_20.view(3, 5, 4, 11776, 32) ;  permute_20 = None
        clone_9 = torch.clone(view_96,memory_format = torch.contiguous_format) ;  view_96 = None
        _unsafe_view_8 = clone_9.view(3, 20, 11776, 32) ;  clone_9 = None
        unbind_int_2 = torch.unbind(_unsafe_view_8) ;  _unsafe_view_8 = None
        getitem_78 = unbind_int_2[0]
        getitem_79 = unbind_int_2[1]
        getitem_80 = unbind_int_2[2];  unbind_int_2 = None
        unsqueeze_30 = torch.unsqueeze(arg60_1,0) ;  arg60_1 = None
        expand_7 = unsqueeze_30.expand(5, -1, -1) ;  unsqueeze_30 = None
        clone_10 = torch.clone(expand_7,memory_format = torch.contiguous_format) ;  expand_7 = None
        _unsafe_view_9 = clone_10.view(20, 1, 32) ;  clone_10 = None
        add_26 = torch.add(getitem_78,_unsafe_view_9) ;  getitem_78 = _unsafe_view_9 = None
        view_97 = add_26.view(20, 368, 32, 32) ;  add_26 = None
        slice_13 = getitem_79[0:] ;  getitem_79 = None
        slice_14 = slice_13[:, :, 0:] ;  slice_13 = None
        index_8 = slice_14[:,remainder] ;  slice_14 = None
        slice_15 = getitem_80[0:] ;  getitem_80 = None
        slice_16 = slice_15[:, :, 0:] ;  slice_15 = None
        index_9 = slice_16[:,remainder] ;  slice_16 = None
        view_98 = masked_fill_3.view(20, 368, 32, 128) ;  masked_fill_3 = None
        expand_8 = view_98.expand(20, 368, 32, 128) ;  view_98 = None
        _scaled_dot_product_efficient_attention_default_1 = (torch.nn.functional.scaled_dot_product_attention(view_97,index_8,index_9,expand_8,False),) ;  view_97 = index_8 = index_9 = expand_8 = None
        getitem_81 = _scaled_dot_product_efficient_attention_default_1[0]
        view_99 = getitem_81.view(5, 4, 368, 32, 32) ;  getitem_81 = None
        permute_21 = view_99.permute(0, 2, 3, 1, 4) ;  view_99 = None
        clone_11 = torch.clone(permute_21,memory_format = torch.contiguous_format) ;  permute_21 = None
        _unsafe_view_10 = clone_11.view(5, 11776, 128) ;  clone_11 = None
        t_26 = arg63_1.t() ;  arg63_1 = None
        clone_12 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_11 = clone_12.view(58880, 128) ;  clone_12 = None
        mm_26 = torch.mm(_unsafe_view_11,t_26) ;  _unsafe_view_11 = t_26 = None
        view_100 = mm_26.view(5, 11776, 128) ;  mm_26 = None
        add_27 = torch.add(view_100,arg64_1) ;  view_100 = arg64_1 = None
        sigmoid_2 = torch.sigmoid(add_27) ;  add_27 = None
        mul_14 = torch.mul(_unsafe_view_10,sigmoid_2) ;  _unsafe_view_10 = sigmoid_2 = None
        native_layer_norm_default_16 = (torch.nn.functional.layer_norm(masked_fill_2,[128],None,None,0.1),) 
        getitem_85 = native_layer_norm_default_16[0]
        t_27 = arg45_1.t() ;  arg45_1 = None
        clone_13 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_12 = clone_13.view(58880, 128) ;  clone_13 = None
        mm_27 = torch.mm(_unsafe_view_12,t_27) ;  _unsafe_view_12 = t_27 = None
        view_101 = mm_27.view(5, 11776, 256) ;  mm_27 = None
        split_tensor_12 = torch.split(view_101,128,dim = -1) ;  view_101 = None
        getitem_88 = split_tensor_12[0]
        getitem_89 = split_tensor_12[1];  split_tensor_12 = None
        add_28 = torch.add(getitem_88,1) ;  getitem_88 = None
        mul_15 = torch.mul(getitem_85,add_28) ;  getitem_85 = add_28 = None
        add_29 = torch.add(mul_15,getitem_89) ;  mul_15 = getitem_89 = None
        t_28 = arg46_1.t() ;  arg46_1 = None
        view_102 = add_29.view(58880, 128) ;  add_29 = None
        mm_28 = torch.mm(view_102,t_28) ;  view_102 = t_28 = None
        view_103 = mm_28.view(5, 11776, 512) ;  mm_28 = None
        split_tensor_13 = torch.split(view_103,256,dim = -1) ;  view_103 = None
        getitem_90 = split_tensor_13[0]
        getitem_91 = split_tensor_13[1];  split_tensor_13 = None
        silu_5 = torch.nn.functional.silu(getitem_90) ;  getitem_90 = None
        mul_16 = torch.mul(silu_5,getitem_91) ;  silu_5 = getitem_91 = None
        t_29 = arg48_1.t() ;  arg48_1 = None
        clone_14 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_13 = clone_14.view(58880, 128) ;  clone_14 = None
        mm_29 = torch.mm(_unsafe_view_13,t_29) ;  _unsafe_view_13 = t_29 = None
        view_104 = mm_29.view(5, 11776, 128) ;  mm_29 = None
        add_30 = torch.add(view_104,arg49_1) ;  view_104 = arg49_1 = None
        sigmoid_3 = torch.sigmoid(add_30) ;  add_30 = None
        t_30 = arg47_1.t() ;  arg47_1 = None
        view_105 = mul_16.view(58880, 256) ;  mul_16 = None
        mm_30 = torch.mm(view_105,t_30) ;  view_105 = t_30 = None
        view_106 = mm_30.view(5, 11776, 128) ;  mm_30 = None
        mul_17 = torch.mul(sigmoid_3,view_106) ;  sigmoid_3 = view_106 = None
        add_31 = torch.add(masked_fill_2,mul_17) ;  masked_fill_2 = mul_17 = None
        add_32 = torch.add(add_31,mul_14) ;  add_31 = mul_14 = None
        unsqueeze_31 = torch.unsqueeze(view_63,-1) ;  view_63 = None
        bitwise_not_4 = torch.bitwise_not(unsqueeze_31) ;  unsqueeze_31 = None
        masked_fill_4 = add_32.masked_fill(bitwise_not_4,0.0) ;  add_32 = bitwise_not_4 = None
        unsqueeze_32 = torch.unsqueeze(getitem_48,5) ;  getitem_48 = None
        permute_22 = unsqueeze_32.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_32 = None
        unsqueeze_33 = torch.unsqueeze(getitem_53,2) ;  getitem_53 = None
        unsqueeze_34 = torch.unsqueeze(unsqueeze_33,3) ;  unsqueeze_33 = None
        unsqueeze_35 = torch.unsqueeze(unsqueeze_34,4) ;  unsqueeze_34 = None
        unsqueeze_36 = torch.unsqueeze(unsqueeze_35,5) ;  unsqueeze_35 = None
        permute_23 = unsqueeze_36.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_36 = None
        permute_24 = permute_22.permute(0, 2, 3, 4, 5, 1) ;  permute_22 = None
        view_107 = permute_24.view(1, 7536640, 16) ;  permute_24 = None
        permute_25 = permute_23.permute(5, 1, 0, 2, 3, 4) ;  permute_23 = None
        view_108 = permute_25.view(1, 16, 4) ;  permute_25 = None
        bmm_4 = torch.bmm(view_107,view_108) ;  view_107 = view_108 = None
        view_109 = bmm_4.view(5, 368, 32, 128, 1, 4) ;  bmm_4 = None
        permute_26 = view_109.permute(0, 5, 1, 2, 3, 4) ;  view_109 = None
        view_110 = permute_26.view(5, 4, 368, 32, 128) ;  permute_26 = None
        view_111 = view_62.view(5, 1, 368, 32, 128) ;  view_62 = None
        bitwise_not_5 = torch.bitwise_not(view_111) ;  view_111 = None
        masked_fill_5 = view_110.masked_fill(bitwise_not_5,-10000) ;  view_110 = bitwise_not_5 = None
        native_layer_norm_default_17 = (torch.nn.functional.layer_norm(masked_fill_4,[128],None,None,0.1),) 
        getitem_92 = native_layer_norm_default_17[0]
        t_31 = arg66_1.t() ;  arg66_1 = None
        clone_15 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_14 = clone_15.view(58880, 128) ;  clone_15 = None
        mm_31 = torch.mm(_unsafe_view_14,t_31) ;  _unsafe_view_14 = t_31 = None
        view_112 = mm_31.view(5, 11776, 256) ;  mm_31 = None
        split_tensor_14 = torch.split(view_112,128,dim = -1) ;  view_112 = None
        getitem_95 = split_tensor_14[0]
        getitem_96 = split_tensor_14[1];  split_tensor_14 = None
        add_33 = torch.add(getitem_95,1) ;  getitem_95 = None
        mul_18 = torch.mul(getitem_92,add_33) ;  getitem_92 = add_33 = None
        add_34 = torch.add(mul_18,getitem_96) ;  mul_18 = getitem_96 = None
        unsqueeze_37 = torch.unsqueeze(add_34,3) ;  add_34 = None
        unsqueeze_38 = torch.unsqueeze(unsqueeze_37,4) ;  unsqueeze_37 = None
        unsqueeze_39 = torch.unsqueeze(unsqueeze_38,5) ;  unsqueeze_38 = None
        permute_27 = unsqueeze_39.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_39 = None
        unsqueeze_40 = torch.unsqueeze(arg67_1,4) ;  arg67_1 = None
        unsqueeze_41 = torch.unsqueeze(unsqueeze_40,5) ;  unsqueeze_40 = None
        permute_28 = unsqueeze_41.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_41 = None
        permute_29 = permute_27.permute(1, 3, 5, 0, 2, 4) ;  permute_27 = None
        view_113 = permute_29.view(1, 58880, 128) ;  permute_29 = None
        permute_30 = permute_28.permute(5, 0, 2, 4, 1, 3) ;  permute_28 = None
        view_114 = permute_30.view(1, 128, 384) ;  permute_30 = None
        bmm_5 = torch.bmm(view_113,view_114) ;  view_113 = view_114 = None
        view_115 = bmm_5.view(5, 11776, 1, 3, 4, 32) ;  bmm_5 = None
        permute_31 = view_115.permute(3, 0, 4, 1, 5, 2) ;  view_115 = None
        view_116 = permute_31.view(3, 5, 4, 11776, 32) ;  permute_31 = None
        clone_16 = torch.clone(view_116,memory_format = torch.contiguous_format) ;  view_116 = None
        _unsafe_view_15 = clone_16.view(3, 20, 11776, 32) ;  clone_16 = None
        unbind_int_3 = torch.unbind(_unsafe_view_15) ;  _unsafe_view_15 = None
        getitem_97 = unbind_int_3[0]
        getitem_98 = unbind_int_3[1]
        getitem_99 = unbind_int_3[2];  unbind_int_3 = None
        unsqueeze_42 = torch.unsqueeze(arg65_1,0) ;  arg65_1 = None
        expand_9 = unsqueeze_42.expand(5, -1, -1) ;  unsqueeze_42 = None
        clone_17 = torch.clone(expand_9,memory_format = torch.contiguous_format) ;  expand_9 = None
        _unsafe_view_16 = clone_17.view(20, 1, 32) ;  clone_17 = None
        add_35 = torch.add(getitem_97,_unsafe_view_16) ;  getitem_97 = _unsafe_view_16 = None
        view_117 = add_35.view(20, 368, 32, 32) ;  add_35 = None
        slice_17 = getitem_98[0:] ;  getitem_98 = None
        slice_18 = slice_17[:, :, 0:] ;  slice_17 = None
        index_10 = slice_18[:,remainder] ;  slice_18 = None
        slice_19 = getitem_99[0:] ;  getitem_99 = None
        slice_20 = slice_19[:, :, 0:] ;  slice_19 = None
        index_11 = slice_20[:,remainder] ;  slice_20 = remainder = None
        view_118 = masked_fill_5.view(20, 368, 32, 128) ;  masked_fill_5 = None
        expand_10 = view_118.expand(20, 368, 32, 128) ;  view_118 = None
        _scaled_dot_product_efficient_attention_default_2 = (torch.nn.functional.scaled_dot_product_attention(view_117,index_10,index_11,expand_10,False),) ;  view_117 = index_10 = index_11 = expand_10 = None
        getitem_100 = _scaled_dot_product_efficient_attention_default_2[0]
        view_119 = getitem_100.view(5, 4, 368, 32, 32) ;  getitem_100 = None
        permute_32 = view_119.permute(0, 2, 3, 1, 4) ;  view_119 = None
        clone_18 = torch.clone(permute_32,memory_format = torch.contiguous_format) ;  permute_32 = None
        _unsafe_view_17 = clone_18.view(5, 11776, 128) ;  clone_18 = None
        t_32 = arg68_1.t() ;  arg68_1 = None
        clone_19 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_18 = clone_19.view(58880, 128) ;  clone_19 = None
        mm_32 = torch.mm(_unsafe_view_18,t_32) ;  _unsafe_view_18 = t_32 = None
        view_120 = mm_32.view(5, 11776, 128) ;  mm_32 = None
        add_36 = torch.add(view_120,arg69_1) ;  view_120 = arg69_1 = None
        sigmoid_4 = torch.sigmoid(add_36) ;  add_36 = None
        mul_19 = torch.mul(_unsafe_view_17,sigmoid_4) ;  _unsafe_view_17 = sigmoid_4 = None
        native_layer_norm_default_18 = (torch.nn.functional.layer_norm(masked_fill_4,[128],None,None,0.1),) 
        getitem_104 = native_layer_norm_default_18[0]
        t_33 = arg50_1.t() ;  arg50_1 = None
        clone_20 = torch.clone(view_60,memory_format = torch.contiguous_format) 
        _unsafe_view_19 = clone_20.view(58880, 128) ;  clone_20 = None
        mm_33 = torch.mm(_unsafe_view_19,t_33) ;  _unsafe_view_19 = t_33 = None
        view_121 = mm_33.view(5, 11776, 256) ;  mm_33 = None
        split_tensor_15 = torch.split(view_121,128,dim = -1) ;  view_121 = None
        getitem_107 = split_tensor_15[0]
        getitem_108 = split_tensor_15[1];  split_tensor_15 = None
        add_37 = torch.add(getitem_107,1) ;  getitem_107 = None
        mul_20 = torch.mul(getitem_104,add_37) ;  getitem_104 = add_37 = None
        add_38 = torch.add(mul_20,getitem_108) ;  mul_20 = getitem_108 = None
        t_34 = arg51_1.t() ;  arg51_1 = None
        view_122 = add_38.view(58880, 128) ;  add_38 = None
        mm_34 = torch.mm(view_122,t_34) ;  view_122 = t_34 = None
        view_123 = mm_34.view(5, 11776, 512) ;  mm_34 = None
        split_tensor_16 = torch.split(view_123,256,dim = -1) ;  view_123 = None
        getitem_109 = split_tensor_16[0]
        getitem_110 = split_tensor_16[1];  split_tensor_16 = None
        silu_6 = torch.nn.functional.silu(getitem_109) ;  getitem_109 = None
        mul_21 = torch.mul(silu_6,getitem_110) ;  silu_6 = getitem_110 = None
        t_35 = arg53_1.t() ;  arg53_1 = None
        clone_21 = torch.clone(view_60,memory_format = torch.contiguous_format) ;  view_60 = None
        _unsafe_view_20 = clone_21.view(58880, 128) ;  clone_21 = None
        mm_35 = torch.mm(_unsafe_view_20,t_35) ;  _unsafe_view_20 = t_35 = None
        view_124 = mm_35.view(5, 11776, 128) ;  mm_35 = None
        add_39 = torch.add(view_124,arg54_1) ;  view_124 = arg54_1 = None
        sigmoid_5 = torch.sigmoid(add_39) ;  add_39 = None
        t_36 = arg52_1.t() ;  arg52_1 = None
        view_125 = mul_21.view(58880, 256) ;  mul_21 = None
        mm_36 = torch.mm(view_125,t_36) ;  view_125 = t_36 = None
        view_126 = mm_36.view(5, 11776, 128) ;  mm_36 = None
        mul_22 = torch.mul(sigmoid_5,view_126) ;  sigmoid_5 = view_126 = None
        add_40 = torch.add(masked_fill_4,mul_22) ;  masked_fill_4 = mul_22 = None
        add_41 = torch.add(add_40,mul_19) ;  add_40 = mul_19 = None
        view_127 = add_41.view(1, 5, 11776, 128) ;  add_41 = None
        t_37 = arg73_1.t() ;  arg73_1 = None
        view_128 = view_127.view(58880, 128) 
        mm_37 = torch.mm(view_128,t_37) ;  view_128 = t_37 = None
        view_129 = mm_37.view(1, 5, 11776, 768) ;  mm_37 = None
        relu_3 = torch.relu(view_129) ;  view_129 = None
        view_130 = relu_3.view(58880, 768) ;  relu_3 = None
        unsqueeze_43 = torch.unsqueeze(arg349_1,1) 
        expand_11 = unsqueeze_43.expand(-1, 5, -1) ;  unsqueeze_43 = None
        view_133 = expand_11.view(5, 11776) ;  expand_11 = None
        unsqueeze_44 = torch.unsqueeze(arg356_1,1) 
        expand_12 = unsqueeze_44.expand(-1, 5, -1) ;  unsqueeze_44 = None
        view_134 = expand_12.view(5, 11776) ;  expand_12 = None
        view_135 = view_130.view(1, 5, 11776, 768) ;  view_130 = None
        view_136 = view_135.view(5, 11776, 768) ;  view_135 = None
        new_zeros = view_136.new_zeros((5,512,768), pin_memory = False)
        new_zeros_1 = view_136.new_zeros((5,512), pin_memory = False)
        unsqueeze_45 = torch.unsqueeze(view_134,2) 
        expand_13 = unsqueeze_45.expand(-1, -1, 768) ;  unsqueeze_45 = None
        unsqueeze_46 = torch.unsqueeze(view_133,-1) 
        mul_23 = torch.mul(view_136,unsqueeze_46) ;  view_136 = unsqueeze_46 = None
        scatter_reduce = torch.scatter_reduce(new_zeros,1,expand_13,mul_23,'sum') ;  new_zeros = expand_13 = mul_23 = None
        _to_copy = view_133.to(dtype = torch.float32) ;  view_133 = None
        scatter_reduce_1 = torch.scatter_reduce(new_zeros_1,1,view_134,_to_copy,'sum') ;  new_zeros_1 = view_134 = _to_copy = None
        unsqueeze_47 = torch.unsqueeze(scatter_reduce_1,-1) ;  scatter_reduce_1 = None
        clamp = torch.clamp(unsqueeze_47,min = 1) ;  unsqueeze_47 = None
        div = torch.div(scatter_reduce,clamp) ;  scatter_reduce = clamp = None
        view_137 = div.view(1, 5, 512, 768) ;  div = None
        t_38 = arg338_1.t() ;  arg338_1 = None
        view_138 = getitem_33.view(2560, 384) 
        mm_38 = torch.mm(view_138,t_38) ;  view_138 = t_38 = None
        view_139 = mm_38.view(1, 5, 512, 768) ;  mm_38 = None
        add_42 = torch.add(view_137,view_139) ;  view_137 = view_139 = None
        view_140 = getitem_36.view(1, 1, 512, 512, 256) ;  getitem_36 = None
        view_141 = arg351_1.view(1, 1, 512, 1) 
        view_142 = arg351_1.view(1, 1, 1, 512) ;  arg351_1 = None
        bitwise_and_1 = torch.bitwise_and(view_141,view_142) ;  view_141 = view_142 = None
        native_layer_norm_default_19 = (torch.nn.functional.layer_norm(view_140,[256],arg87_1,arg88_1,1e-05),) ;  arg87_1 = arg88_1 = None
        getitem_111 = native_layer_norm_default_19[0]
        t_39 = arg89_1.t() ;  arg89_1 = None
        view_144 = getitem_111.view(262144, 256) ;  getitem_111 = None
        mm_39 = torch.mm(view_144,t_39) ;  view_144 = t_39 = None
        view_145 = mm_39.view(1, 1, 512, 512, 16) ;  mm_39 = None
        native_layer_norm_default_20 = (torch.nn.functional.layer_norm(add_42,[768],None,None,0.1),) 
        getitem_114 = native_layer_norm_default_20[0]
        t_40 = arg83_1.t() ;  arg83_1 = None
        view_146 = getitem_33.view(2560, 384) 
        mm_40 = torch.mm(view_146,t_40) ;  view_146 = t_40 = None
        view_147 = mm_40.view(1, 5, 512, 1536) ;  mm_40 = None
        split_tensor_17 = torch.split(view_147,768,dim = -1) ;  view_147 = None
        getitem_117 = split_tensor_17[0]
        getitem_118 = split_tensor_17[1];  split_tensor_17 = None
        add_43 = torch.add(getitem_117,1) ;  getitem_117 = None
        mul_24 = torch.mul(getitem_114,add_43) ;  getitem_114 = add_43 = None
        add_44 = torch.add(mul_24,getitem_118) ;  mul_24 = getitem_118 = None
        t_41 = arg84_1.t() ;  arg84_1 = None
        view_148 = add_44.view(2560, 768) ;  add_44 = None
        mm_41 = torch.mm(view_148,t_41) ;  view_148 = t_41 = None
        view_149 = mm_41.view(1, 5, 512, 2304) ;  mm_41 = None
        view_150 = view_149.view(1, 5, 512, 16, 144) ;  view_149 = None
        permute_33 = view_150.permute(0, 3, 1, 2, 4) ;  view_150 = None
        split_tensor_18 = torch.split(permute_33,48,dim = -1) ;  permute_33 = None
        getitem_119 = split_tensor_18[0]
        getitem_120 = split_tensor_18[1]
        getitem_121 = split_tensor_18[2];  split_tensor_18 = None
        view_151 = arg77_1.view(1, 16, 1, 1, 48) ;  arg77_1 = None
        add_45 = torch.add(getitem_119,view_151) ;  getitem_119 = view_151 = None
        view_152 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_6 = torch.bitwise_not(view_152) ;  view_152 = None
        masked_fill_6 = view_145.masked_fill(bitwise_not_6,-10000) ;  view_145 = bitwise_not_6 = None
        permute_34 = masked_fill_6.permute(0, 4, 1, 2, 3) ;  masked_fill_6 = None
        view_153 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_7 = torch.bitwise_not(view_153) ;  view_153 = None
        masked_fill_7 = permute_34.masked_fill(bitwise_not_7,-10000) ;  permute_34 = bitwise_not_7 = None
        mul_25 = torch.mul(add_45,0.3799178428257963) ;  add_45 = None
        transpose = torch.transpose(getitem_120,-2,-1) ;  getitem_120 = None
        mul_26 = torch.mul(transpose,0.3799178428257963) ;  transpose = None
        expand_14 = mul_25.expand(1, 16, 5, 512, 48) ;  mul_25 = None
        clone_22 = torch.clone(expand_14,memory_format = torch.contiguous_format) ;  expand_14 = None
        _unsafe_view_21 = clone_22.view(80, 512, 48) ;  clone_22 = None
        expand_15 = mul_26.expand(1, 16, 5, 48, 512) ;  mul_26 = None
        clone_23 = torch.clone(expand_15,memory_format = torch.contiguous_format) ;  expand_15 = None
        _unsafe_view_22 = clone_23.view(80, 48, 512) ;  clone_23 = None
        bmm_6 = torch.bmm(_unsafe_view_21,_unsafe_view_22) ;  _unsafe_view_21 = _unsafe_view_22 = None
        view_154 = bmm_6.view(1, 16, 5, 512, 512) ;  bmm_6 = None
        add_46 = torch.add(view_154,masked_fill_7) ;  view_154 = masked_fill_7 = None
        _softmax = torch.softmax(add_46, -1)
        expand_16 = _softmax.expand(1, 16, 5, 512, 512) ;  _softmax = None
        view_155 = expand_16.view(80, 512, 512) ;  expand_16 = None
        expand_17 = getitem_121.expand(1, 16, 5, 512, 48) ;  getitem_121 = None
        clone_24 = torch.clone(expand_17,memory_format = torch.contiguous_format) ;  expand_17 = None
        _unsafe_view_23 = clone_24.view(80, 512, 48) ;  clone_24 = None
        bmm_7 = torch.bmm(view_155,_unsafe_view_23) ;  view_155 = _unsafe_view_23 = None
        view_156 = bmm_7.view(1, 16, 5, 512, 48) ;  bmm_7 = None
        permute_35 = view_156.permute(0, 2, 3, 1, 4) ;  view_156 = None
        clone_25 = torch.clone(permute_35,memory_format = torch.contiguous_format) ;  permute_35 = None
        _unsafe_view_24 = clone_25.view(1, 5, 512, 768) ;  clone_25 = None
        t_42 = arg90_1.t() ;  arg90_1 = None
        view_157 = _unsafe_view_24.view(2560, 768) ;  _unsafe_view_24 = None
        mm_42 = torch.mm(view_157,t_42) ;  view_157 = t_42 = None
        view_158 = mm_42.view(1, 5, 512, 768) ;  mm_42 = None
        view_159 = getitem_33.view(2560, 384) 
        t_43 = arg85_1.t() ;  arg85_1 = None
        addmm = torch.addmm(arg86_1,view_159,t_43) ;  arg86_1 = view_159 = t_43 = None
        view_160 = addmm.view(1, 5, 512, 768) ;  addmm = None
        sigmoid_6 = torch.sigmoid(view_160) ;  view_160 = None
        mul_27 = torch.mul(sigmoid_6,view_158) ;  sigmoid_6 = view_158 = None
        native_layer_norm_default_21 = (torch.nn.functional.layer_norm(add_42,[768],None,None,0.1),) 
        getitem_122 = native_layer_norm_default_21[0]
        t_44 = arg78_1.t() ;  arg78_1 = None
        view_161 = getitem_33.view(2560, 384) 
        mm_43 = torch.mm(view_161,t_44) ;  view_161 = t_44 = None
        view_162 = mm_43.view(1, 5, 512, 1536) ;  mm_43 = None
        split_tensor_19 = torch.split(view_162,768,dim = -1) ;  view_162 = None
        getitem_125 = split_tensor_19[0]
        getitem_126 = split_tensor_19[1];  split_tensor_19 = None
        add_47 = torch.add(getitem_125,1) ;  getitem_125 = None
        mul_28 = torch.mul(getitem_122,add_47) ;  getitem_122 = add_47 = None
        add_48 = torch.add(mul_28,getitem_126) ;  mul_28 = getitem_126 = None
        t_45 = arg79_1.t() ;  arg79_1 = None
        view_163 = add_48.view(2560, 768) ;  add_48 = None
        mm_44 = torch.mm(view_163,t_45) ;  view_163 = t_45 = None
        view_164 = mm_44.view(1, 5, 512, 3072) ;  mm_44 = None
        split_tensor_20 = torch.split(view_164,1536,dim = -1) ;  view_164 = None
        getitem_127 = split_tensor_20[0]
        getitem_128 = split_tensor_20[1];  split_tensor_20 = None
        silu_7 = torch.nn.functional.silu(getitem_127) ;  getitem_127 = None
        mul_29 = torch.mul(silu_7,getitem_128) ;  silu_7 = getitem_128 = None
        view_165 = getitem_33.view(2560, 384) 
        t_46 = arg81_1.t() ;  arg81_1 = None
        addmm_1 = torch.addmm(arg82_1,view_165,t_46) ;  arg82_1 = view_165 = t_46 = None
        view_166 = addmm_1.view(1, 5, 512, 768) ;  addmm_1 = None
        sigmoid_7 = torch.sigmoid(view_166) ;  view_166 = None
        t_47 = arg80_1.t() ;  arg80_1 = None
        view_167 = mul_29.view(2560, 1536) ;  mul_29 = None
        mm_45 = torch.mm(view_167,t_47) ;  view_167 = t_47 = None
        view_168 = mm_45.view(1, 5, 512, 768) ;  mm_45 = None
        mul_30 = torch.mul(sigmoid_7,view_168) ;  sigmoid_7 = view_168 = None
        add_49 = torch.add(mul_27,mul_30) ;  mul_27 = mul_30 = None
        add_50 = torch.add(add_42,add_49) ;  add_42 = add_49 = None
        native_layer_norm_default_22 = (torch.nn.functional.layer_norm(view_140,[256],arg101_1,arg102_1,1e-05),) ;  arg101_1 = arg102_1 = None
        getitem_129 = native_layer_norm_default_22[0]
        t_48 = arg103_1.t() ;  arg103_1 = None
        view_169 = getitem_129.view(262144, 256) ;  getitem_129 = None
        mm_46 = torch.mm(view_169,t_48) ;  view_169 = t_48 = None
        view_170 = mm_46.view(1, 1, 512, 512, 16) ;  mm_46 = None
        native_layer_norm_default_23 = (torch.nn.functional.layer_norm(add_50,[768],None,None,0.1),) 
        getitem_132 = native_layer_norm_default_23[0]
        t_49 = arg97_1.t() ;  arg97_1 = None
        view_171 = getitem_33.view(2560, 384) 
        mm_47 = torch.mm(view_171,t_49) ;  view_171 = t_49 = None
        view_172 = mm_47.view(1, 5, 512, 1536) ;  mm_47 = None
        split_tensor_21 = torch.split(view_172,768,dim = -1) ;  view_172 = None
        getitem_135 = split_tensor_21[0]
        getitem_136 = split_tensor_21[1];  split_tensor_21 = None
        add_51 = torch.add(getitem_135,1) ;  getitem_135 = None
        mul_31 = torch.mul(getitem_132,add_51) ;  getitem_132 = add_51 = None
        add_52 = torch.add(mul_31,getitem_136) ;  mul_31 = getitem_136 = None
        t_50 = arg98_1.t() ;  arg98_1 = None
        view_173 = add_52.view(2560, 768) ;  add_52 = None
        mm_48 = torch.mm(view_173,t_50) ;  view_173 = t_50 = None
        view_174 = mm_48.view(1, 5, 512, 2304) ;  mm_48 = None
        view_175 = view_174.view(1, 5, 512, 16, 144) ;  view_174 = None
        permute_36 = view_175.permute(0, 3, 1, 2, 4) ;  view_175 = None
        split_tensor_22 = torch.split(permute_36,48,dim = -1) ;  permute_36 = None
        getitem_137 = split_tensor_22[0]
        getitem_138 = split_tensor_22[1]
        getitem_139 = split_tensor_22[2];  split_tensor_22 = None
        view_176 = arg91_1.view(1, 16, 1, 1, 48) ;  arg91_1 = None
        add_53 = torch.add(getitem_137,view_176) ;  getitem_137 = view_176 = None
        view_177 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_8 = torch.bitwise_not(view_177) ;  view_177 = None
        masked_fill_8 = view_170.masked_fill(bitwise_not_8,-10000) ;  view_170 = bitwise_not_8 = None
        permute_37 = masked_fill_8.permute(0, 4, 1, 2, 3) ;  masked_fill_8 = None
        view_178 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_9 = torch.bitwise_not(view_178) ;  view_178 = None
        masked_fill_9 = permute_37.masked_fill(bitwise_not_9,-10000) ;  permute_37 = bitwise_not_9 = None
        mul_32 = torch.mul(add_53,0.3799178428257963) ;  add_53 = None
        transpose_1 = torch.transpose(getitem_138,-2,-1) ;  getitem_138 = None
        mul_33 = torch.mul(transpose_1,0.3799178428257963) ;  transpose_1 = None
        expand_18 = mul_32.expand(1, 16, 5, 512, 48) ;  mul_32 = None
        clone_26 = torch.clone(expand_18,memory_format = torch.contiguous_format) ;  expand_18 = None
        _unsafe_view_25 = clone_26.view(80, 512, 48) ;  clone_26 = None
        expand_19 = mul_33.expand(1, 16, 5, 48, 512) ;  mul_33 = None
        clone_27 = torch.clone(expand_19,memory_format = torch.contiguous_format) ;  expand_19 = None
        _unsafe_view_26 = clone_27.view(80, 48, 512) ;  clone_27 = None
        bmm_8 = torch.bmm(_unsafe_view_25,_unsafe_view_26) ;  _unsafe_view_25 = _unsafe_view_26 = None
        view_179 = bmm_8.view(1, 16, 5, 512, 512) ;  bmm_8 = None
        add_54 = torch.add(view_179,masked_fill_9) ;  view_179 = masked_fill_9 = None
        _softmax_1 = torch.softmax(add_54, -1)
        expand_20 = _softmax_1.expand(1, 16, 5, 512, 512) ;  _softmax_1 = None
        view_180 = expand_20.view(80, 512, 512) ;  expand_20 = None
        expand_21 = getitem_139.expand(1, 16, 5, 512, 48) ;  getitem_139 = None
        clone_28 = torch.clone(expand_21,memory_format = torch.contiguous_format) ;  expand_21 = None
        _unsafe_view_27 = clone_28.view(80, 512, 48) ;  clone_28 = None
        bmm_9 = torch.bmm(view_180,_unsafe_view_27) ;  view_180 = _unsafe_view_27 = None
        view_181 = bmm_9.view(1, 16, 5, 512, 48) ;  bmm_9 = None
        permute_38 = view_181.permute(0, 2, 3, 1, 4) ;  view_181 = None
        clone_29 = torch.clone(permute_38,memory_format = torch.contiguous_format) ;  permute_38 = None
        _unsafe_view_28 = clone_29.view(1, 5, 512, 768) ;  clone_29 = None
        t_51 = arg104_1.t() ;  arg104_1 = None
        view_182 = _unsafe_view_28.view(2560, 768) ;  _unsafe_view_28 = None
        mm_49 = torch.mm(view_182,t_51) ;  view_182 = t_51 = None
        view_183 = mm_49.view(1, 5, 512, 768) ;  mm_49 = None
        view_184 = getitem_33.view(2560, 384) 
        t_52 = arg99_1.t() ;  arg99_1 = None
        addmm_2 = torch.addmm(arg100_1,view_184,t_52) ;  arg100_1 = view_184 = t_52 = None
        view_185 = addmm_2.view(1, 5, 512, 768) ;  addmm_2 = None
        sigmoid_8 = torch.sigmoid(view_185) ;  view_185 = None
        mul_34 = torch.mul(sigmoid_8,view_183) ;  sigmoid_8 = view_183 = None
        native_layer_norm_default_24 = (torch.nn.functional.layer_norm(add_50,[768],None,None,0.1),) 
        getitem_140 = native_layer_norm_default_24[0]
        t_53 = arg92_1.t() ;  arg92_1 = None
        view_186 = getitem_33.view(2560, 384) 
        mm_50 = torch.mm(view_186,t_53) ;  view_186 = t_53 = None
        view_187 = mm_50.view(1, 5, 512, 1536) ;  mm_50 = None
        split_tensor_23 = torch.split(view_187,768,dim = -1) ;  view_187 = None
        getitem_143 = split_tensor_23[0]
        getitem_144 = split_tensor_23[1];  split_tensor_23 = None
        add_55 = torch.add(getitem_143,1) ;  getitem_143 = None
        mul_35 = torch.mul(getitem_140,add_55) ;  getitem_140 = add_55 = None
        add_56 = torch.add(mul_35,getitem_144) ;  mul_35 = getitem_144 = None
        t_54 = arg93_1.t() ;  arg93_1 = None
        view_188 = add_56.view(2560, 768) ;  add_56 = None
        mm_51 = torch.mm(view_188,t_54) ;  view_188 = t_54 = None
        view_189 = mm_51.view(1, 5, 512, 3072) ;  mm_51 = None
        split_tensor_24 = torch.split(view_189,1536,dim = -1) ;  view_189 = None
        getitem_145 = split_tensor_24[0]
        getitem_146 = split_tensor_24[1];  split_tensor_24 = None
        silu_8 = torch.nn.functional.silu(getitem_145) ;  getitem_145 = None
        mul_36 = torch.mul(silu_8,getitem_146) ;  silu_8 = getitem_146 = None
        view_190 = getitem_33.view(2560, 384) 
        t_55 = arg95_1.t() ;  arg95_1 = None
        addmm_3 = torch.addmm(arg96_1,view_190,t_55) ;  arg96_1 = view_190 = t_55 = None
        view_191 = addmm_3.view(1, 5, 512, 768) ;  addmm_3 = None
        sigmoid_9 = torch.sigmoid(view_191) ;  view_191 = None
        t_56 = arg94_1.t() ;  arg94_1 = None
        view_192 = mul_36.view(2560, 1536) ;  mul_36 = None
        mm_52 = torch.mm(view_192,t_56) ;  view_192 = t_56 = None
        view_193 = mm_52.view(1, 5, 512, 768) ;  mm_52 = None
        mul_37 = torch.mul(sigmoid_9,view_193) ;  sigmoid_9 = view_193 = None
        add_57 = torch.add(mul_34,mul_37) ;  mul_34 = mul_37 = None
        add_58 = torch.add(add_50,add_57) ;  add_50 = add_57 = None
        native_layer_norm_default_25 = (torch.nn.functional.layer_norm(view_140,[256],arg115_1,arg116_1,1e-05),) ;  arg115_1 = arg116_1 = None
        getitem_147 = native_layer_norm_default_25[0]
        t_57 = arg117_1.t() ;  arg117_1 = None
        view_194 = getitem_147.view(262144, 256) ;  getitem_147 = None
        mm_53 = torch.mm(view_194,t_57) ;  view_194 = t_57 = None
        view_195 = mm_53.view(1, 1, 512, 512, 16) ;  mm_53 = None
        native_layer_norm_default_26 = (torch.nn.functional.layer_norm(add_58,[768],None,None,0.1),) 
        getitem_150 = native_layer_norm_default_26[0]
        t_58 = arg111_1.t() ;  arg111_1 = None
        view_196 = getitem_33.view(2560, 384) 
        mm_54 = torch.mm(view_196,t_58) ;  view_196 = t_58 = None
        view_197 = mm_54.view(1, 5, 512, 1536) ;  mm_54 = None
        split_tensor_25 = torch.split(view_197,768,dim = -1) ;  view_197 = None
        getitem_153 = split_tensor_25[0]
        getitem_154 = split_tensor_25[1];  split_tensor_25 = None
        add_59 = torch.add(getitem_153,1) ;  getitem_153 = None
        mul_38 = torch.mul(getitem_150,add_59) ;  getitem_150 = add_59 = None
        add_60 = torch.add(mul_38,getitem_154) ;  mul_38 = getitem_154 = None
        t_59 = arg112_1.t() ;  arg112_1 = None
        view_198 = add_60.view(2560, 768) ;  add_60 = None
        mm_55 = torch.mm(view_198,t_59) ;  view_198 = t_59 = None
        view_199 = mm_55.view(1, 5, 512, 2304) ;  mm_55 = None
        view_200 = view_199.view(1, 5, 512, 16, 144) ;  view_199 = None
        permute_39 = view_200.permute(0, 3, 1, 2, 4) ;  view_200 = None
        split_tensor_26 = torch.split(permute_39,48,dim = -1) ;  permute_39 = None
        getitem_155 = split_tensor_26[0]
        getitem_156 = split_tensor_26[1]
        getitem_157 = split_tensor_26[2];  split_tensor_26 = None
        view_201 = arg105_1.view(1, 16, 1, 1, 48) ;  arg105_1 = None
        add_61 = torch.add(getitem_155,view_201) ;  getitem_155 = view_201 = None
        view_202 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_10 = torch.bitwise_not(view_202) ;  view_202 = None
        masked_fill_10 = view_195.masked_fill(bitwise_not_10,-10000) ;  view_195 = bitwise_not_10 = None
        permute_40 = masked_fill_10.permute(0, 4, 1, 2, 3) ;  masked_fill_10 = None
        view_203 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_11 = torch.bitwise_not(view_203) ;  view_203 = None
        masked_fill_11 = permute_40.masked_fill(bitwise_not_11,-10000) ;  permute_40 = bitwise_not_11 = None
        mul_39 = torch.mul(add_61,0.3799178428257963) ;  add_61 = None
        transpose_2 = torch.transpose(getitem_156,-2,-1) ;  getitem_156 = None
        mul_40 = torch.mul(transpose_2,0.3799178428257963) ;  transpose_2 = None
        expand_22 = mul_39.expand(1, 16, 5, 512, 48) ;  mul_39 = None
        clone_30 = torch.clone(expand_22,memory_format = torch.contiguous_format) ;  expand_22 = None
        _unsafe_view_29 = clone_30.view(80, 512, 48) ;  clone_30 = None
        expand_23 = mul_40.expand(1, 16, 5, 48, 512) ;  mul_40 = None
        clone_31 = torch.clone(expand_23,memory_format = torch.contiguous_format) ;  expand_23 = None
        _unsafe_view_30 = clone_31.view(80, 48, 512) ;  clone_31 = None
        bmm_10 = torch.bmm(_unsafe_view_29,_unsafe_view_30) ;  _unsafe_view_29 = _unsafe_view_30 = None
        view_204 = bmm_10.view(1, 16, 5, 512, 512) ;  bmm_10 = None
        add_62 = torch.add(view_204,masked_fill_11) ;  view_204 = masked_fill_11 = None
        _softmax_2 = torch.softmax(add_62, -1)
        expand_24 = _softmax_2.expand(1, 16, 5, 512, 512) ;  _softmax_2 = None
        view_205 = expand_24.view(80, 512, 512) ;  expand_24 = None
        expand_25 = getitem_157.expand(1, 16, 5, 512, 48) ;  getitem_157 = None
        clone_32 = torch.clone(expand_25,memory_format = torch.contiguous_format) ;  expand_25 = None
        _unsafe_view_31 = clone_32.view(80, 512, 48) ;  clone_32 = None
        bmm_11 = torch.bmm(view_205,_unsafe_view_31) ;  view_205 = _unsafe_view_31 = None
        view_206 = bmm_11.view(1, 16, 5, 512, 48) ;  bmm_11 = None
        permute_41 = view_206.permute(0, 2, 3, 1, 4) ;  view_206 = None
        clone_33 = torch.clone(permute_41,memory_format = torch.contiguous_format) ;  permute_41 = None
        _unsafe_view_32 = clone_33.view(1, 5, 512, 768) ;  clone_33 = None
        t_60 = arg118_1.t() ;  arg118_1 = None
        view_207 = _unsafe_view_32.view(2560, 768) ;  _unsafe_view_32 = None
        mm_56 = torch.mm(view_207,t_60) ;  view_207 = t_60 = None
        view_208 = mm_56.view(1, 5, 512, 768) ;  mm_56 = None
        view_209 = getitem_33.view(2560, 384) 
        t_61 = arg113_1.t() ;  arg113_1 = None
        addmm_4 = torch.addmm(arg114_1,view_209,t_61) ;  arg114_1 = view_209 = t_61 = None
        view_210 = addmm_4.view(1, 5, 512, 768) ;  addmm_4 = None
        sigmoid_10 = torch.sigmoid(view_210) ;  view_210 = None
        mul_41 = torch.mul(sigmoid_10,view_208) ;  sigmoid_10 = view_208 = None
        native_layer_norm_default_27 = (torch.nn.functional.layer_norm(add_58,[768],None,None,0.1),) 
        getitem_158 = native_layer_norm_default_27[0]
        t_62 = arg106_1.t() ;  arg106_1 = None
        view_211 = getitem_33.view(2560, 384) 
        mm_57 = torch.mm(view_211,t_62) ;  view_211 = t_62 = None
        view_212 = mm_57.view(1, 5, 512, 1536) ;  mm_57 = None
        split_tensor_27 = torch.split(view_212,768,dim = -1) ;  view_212 = None
        getitem_161 = split_tensor_27[0]
        getitem_162 = split_tensor_27[1];  split_tensor_27 = None
        add_63 = torch.add(getitem_161,1) ;  getitem_161 = None
        mul_42 = torch.mul(getitem_158,add_63) ;  getitem_158 = add_63 = None
        add_64 = torch.add(mul_42,getitem_162) ;  mul_42 = getitem_162 = None
        t_63 = arg107_1.t() ;  arg107_1 = None
        view_213 = add_64.view(2560, 768) ;  add_64 = None
        mm_58 = torch.mm(view_213,t_63) ;  view_213 = t_63 = None
        view_214 = mm_58.view(1, 5, 512, 3072) ;  mm_58 = None
        split_tensor_28 = torch.split(view_214,1536,dim = -1) ;  view_214 = None
        getitem_163 = split_tensor_28[0]
        getitem_164 = split_tensor_28[1];  split_tensor_28 = None
        silu_9 = torch.nn.functional.silu(getitem_163) ;  getitem_163 = None
        mul_43 = torch.mul(silu_9,getitem_164) ;  silu_9 = getitem_164 = None
        view_215 = getitem_33.view(2560, 384) 
        t_64 = arg109_1.t() ;  arg109_1 = None
        addmm_5 = torch.addmm(arg110_1,view_215,t_64) ;  arg110_1 = view_215 = t_64 = None
        view_216 = addmm_5.view(1, 5, 512, 768) ;  addmm_5 = None
        sigmoid_11 = torch.sigmoid(view_216) ;  view_216 = None
        t_65 = arg108_1.t() ;  arg108_1 = None
        view_217 = mul_43.view(2560, 1536) ;  mul_43 = None
        mm_59 = torch.mm(view_217,t_65) ;  view_217 = t_65 = None
        view_218 = mm_59.view(1, 5, 512, 768) ;  mm_59 = None
        mul_44 = torch.mul(sigmoid_11,view_218) ;  sigmoid_11 = view_218 = None
        add_65 = torch.add(mul_41,mul_44) ;  mul_41 = mul_44 = None
        add_66 = torch.add(add_58,add_65) ;  add_58 = add_65 = None
        native_layer_norm_default_28 = (torch.nn.functional.layer_norm(view_140,[256],arg129_1,arg130_1,1e-05),) ;  arg129_1 = arg130_1 = None
        getitem_165 = native_layer_norm_default_28[0]
        t_66 = arg131_1.t() ;  arg131_1 = None
        view_219 = getitem_165.view(262144, 256) ;  getitem_165 = None
        mm_60 = torch.mm(view_219,t_66) ;  view_219 = t_66 = None
        view_220 = mm_60.view(1, 1, 512, 512, 16) ;  mm_60 = None
        native_layer_norm_default_29 = (torch.nn.functional.layer_norm(add_66,[768],None,None,0.1),) 
        getitem_168 = native_layer_norm_default_29[0]
        t_67 = arg125_1.t() ;  arg125_1 = None
        view_221 = getitem_33.view(2560, 384) 
        mm_61 = torch.mm(view_221,t_67) ;  view_221 = t_67 = None
        view_222 = mm_61.view(1, 5, 512, 1536) ;  mm_61 = None
        split_tensor_29 = torch.split(view_222,768,dim = -1) ;  view_222 = None
        getitem_171 = split_tensor_29[0]
        getitem_172 = split_tensor_29[1];  split_tensor_29 = None
        add_67 = torch.add(getitem_171,1) ;  getitem_171 = None
        mul_45 = torch.mul(getitem_168,add_67) ;  getitem_168 = add_67 = None
        add_68 = torch.add(mul_45,getitem_172) ;  mul_45 = getitem_172 = None
        t_68 = arg126_1.t() ;  arg126_1 = None
        view_223 = add_68.view(2560, 768) ;  add_68 = None
        mm_62 = torch.mm(view_223,t_68) ;  view_223 = t_68 = None
        view_224 = mm_62.view(1, 5, 512, 2304) ;  mm_62 = None
        view_225 = view_224.view(1, 5, 512, 16, 144) ;  view_224 = None
        permute_42 = view_225.permute(0, 3, 1, 2, 4) ;  view_225 = None
        split_tensor_30 = torch.split(permute_42,48,dim = -1) ;  permute_42 = None
        getitem_173 = split_tensor_30[0]
        getitem_174 = split_tensor_30[1]
        getitem_175 = split_tensor_30[2];  split_tensor_30 = None
        view_226 = arg119_1.view(1, 16, 1, 1, 48) ;  arg119_1 = None
        add_69 = torch.add(getitem_173,view_226) ;  getitem_173 = view_226 = None
        view_227 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_12 = torch.bitwise_not(view_227) ;  view_227 = None
        masked_fill_12 = view_220.masked_fill(bitwise_not_12,-10000) ;  view_220 = bitwise_not_12 = None
        permute_43 = masked_fill_12.permute(0, 4, 1, 2, 3) ;  masked_fill_12 = None
        view_228 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_13 = torch.bitwise_not(view_228) ;  view_228 = None
        masked_fill_13 = permute_43.masked_fill(bitwise_not_13,-10000) ;  permute_43 = bitwise_not_13 = None
        mul_46 = torch.mul(add_69,0.3799178428257963) ;  add_69 = None
        transpose_3 = torch.transpose(getitem_174,-2,-1) ;  getitem_174 = None
        mul_47 = torch.mul(transpose_3,0.3799178428257963) ;  transpose_3 = None
        expand_26 = mul_46.expand(1, 16, 5, 512, 48) ;  mul_46 = None
        clone_34 = torch.clone(expand_26,memory_format = torch.contiguous_format) ;  expand_26 = None
        _unsafe_view_33 = clone_34.view(80, 512, 48) ;  clone_34 = None
        expand_27 = mul_47.expand(1, 16, 5, 48, 512) ;  mul_47 = None
        clone_35 = torch.clone(expand_27,memory_format = torch.contiguous_format) ;  expand_27 = None
        _unsafe_view_34 = clone_35.view(80, 48, 512) ;  clone_35 = None
        bmm_12 = torch.bmm(_unsafe_view_33,_unsafe_view_34) ;  _unsafe_view_33 = _unsafe_view_34 = None
        view_229 = bmm_12.view(1, 16, 5, 512, 512) ;  bmm_12 = None
        add_70 = torch.add(view_229,masked_fill_13) ;  view_229 = masked_fill_13 = None
        _softmax_3 = torch.softmax(add_70, -1)
        expand_28 = _softmax_3.expand(1, 16, 5, 512, 512) ;  _softmax_3 = None
        view_230 = expand_28.view(80, 512, 512) ;  expand_28 = None
        expand_29 = getitem_175.expand(1, 16, 5, 512, 48) ;  getitem_175 = None
        clone_36 = torch.clone(expand_29,memory_format = torch.contiguous_format) ;  expand_29 = None
        _unsafe_view_35 = clone_36.view(80, 512, 48) ;  clone_36 = None
        bmm_13 = torch.bmm(view_230,_unsafe_view_35) ;  view_230 = _unsafe_view_35 = None
        view_231 = bmm_13.view(1, 16, 5, 512, 48) ;  bmm_13 = None
        permute_44 = view_231.permute(0, 2, 3, 1, 4) ;  view_231 = None
        clone_37 = torch.clone(permute_44,memory_format = torch.contiguous_format) ;  permute_44 = None
        _unsafe_view_36 = clone_37.view(1, 5, 512, 768) ;  clone_37 = None
        t_69 = arg132_1.t() ;  arg132_1 = None
        view_232 = _unsafe_view_36.view(2560, 768) ;  _unsafe_view_36 = None
        mm_63 = torch.mm(view_232,t_69) ;  view_232 = t_69 = None
        view_233 = mm_63.view(1, 5, 512, 768) ;  mm_63 = None
        view_234 = getitem_33.view(2560, 384) 
        t_70 = arg127_1.t() ;  arg127_1 = None
        addmm_6 = torch.addmm(arg128_1,view_234,t_70) ;  arg128_1 = view_234 = t_70 = None
        view_235 = addmm_6.view(1, 5, 512, 768) ;  addmm_6 = None
        sigmoid_12 = torch.sigmoid(view_235) ;  view_235 = None
        mul_48 = torch.mul(sigmoid_12,view_233) ;  sigmoid_12 = view_233 = None
        native_layer_norm_default_30 = (torch.nn.functional.layer_norm(add_66,[768],None,None,0.1),) 
        getitem_176 = native_layer_norm_default_30[0]
        t_71 = arg120_1.t() ;  arg120_1 = None
        view_236 = getitem_33.view(2560, 384) 
        mm_64 = torch.mm(view_236,t_71) ;  view_236 = t_71 = None
        view_237 = mm_64.view(1, 5, 512, 1536) ;  mm_64 = None
        split_tensor_31 = torch.split(view_237,768,dim = -1) ;  view_237 = None
        getitem_179 = split_tensor_31[0]
        getitem_180 = split_tensor_31[1];  split_tensor_31 = None
        add_71 = torch.add(getitem_179,1) ;  getitem_179 = None
        mul_49 = torch.mul(getitem_176,add_71) ;  getitem_176 = add_71 = None
        add_72 = torch.add(mul_49,getitem_180) ;  mul_49 = getitem_180 = None
        t_72 = arg121_1.t() ;  arg121_1 = None
        view_238 = add_72.view(2560, 768) ;  add_72 = None
        mm_65 = torch.mm(view_238,t_72) ;  view_238 = t_72 = None
        view_239 = mm_65.view(1, 5, 512, 3072) ;  mm_65 = None
        split_tensor_32 = torch.split(view_239,1536,dim = -1) ;  view_239 = None
        getitem_181 = split_tensor_32[0]
        getitem_182 = split_tensor_32[1];  split_tensor_32 = None
        silu_10 = torch.nn.functional.silu(getitem_181) ;  getitem_181 = None
        mul_50 = torch.mul(silu_10,getitem_182) ;  silu_10 = getitem_182 = None
        view_240 = getitem_33.view(2560, 384) 
        t_73 = arg123_1.t() ;  arg123_1 = None
        addmm_7 = torch.addmm(arg124_1,view_240,t_73) ;  arg124_1 = view_240 = t_73 = None
        view_241 = addmm_7.view(1, 5, 512, 768) ;  addmm_7 = None
        sigmoid_13 = torch.sigmoid(view_241) ;  view_241 = None
        t_74 = arg122_1.t() ;  arg122_1 = None
        view_242 = mul_50.view(2560, 1536) ;  mul_50 = None
        mm_66 = torch.mm(view_242,t_74) ;  view_242 = t_74 = None
        view_243 = mm_66.view(1, 5, 512, 768) ;  mm_66 = None
        mul_51 = torch.mul(sigmoid_13,view_243) ;  sigmoid_13 = view_243 = None
        add_73 = torch.add(mul_48,mul_51) ;  mul_48 = mul_51 = None
        add_74 = torch.add(add_66,add_73) ;  add_66 = add_73 = None
        native_layer_norm_default_31 = (torch.nn.functional.layer_norm(view_140,[256],arg143_1,arg144_1,1e-05),) ;  arg143_1 = arg144_1 = None
        getitem_183 = native_layer_norm_default_31[0]
        t_75 = arg145_1.t() ;  arg145_1 = None
        view_244 = getitem_183.view(262144, 256) ;  getitem_183 = None
        mm_67 = torch.mm(view_244,t_75) ;  view_244 = t_75 = None
        view_245 = mm_67.view(1, 1, 512, 512, 16) ;  mm_67 = None
        native_layer_norm_default_32 = (torch.nn.functional.layer_norm(add_74,[768],None,None,0.1),) 
        getitem_186 = native_layer_norm_default_32[0]
        t_76 = arg139_1.t() ;  arg139_1 = None
        view_246 = getitem_33.view(2560, 384) 
        mm_68 = torch.mm(view_246,t_76) ;  view_246 = t_76 = None
        view_247 = mm_68.view(1, 5, 512, 1536) ;  mm_68 = None
        split_tensor_33 = torch.split(view_247,768,dim = -1) ;  view_247 = None
        getitem_189 = split_tensor_33[0]
        getitem_190 = split_tensor_33[1];  split_tensor_33 = None
        add_75 = torch.add(getitem_189,1) ;  getitem_189 = None
        mul_52 = torch.mul(getitem_186,add_75) ;  getitem_186 = add_75 = None
        add_76 = torch.add(mul_52,getitem_190) ;  mul_52 = getitem_190 = None
        t_77 = arg140_1.t() ;  arg140_1 = None
        view_248 = add_76.view(2560, 768) ;  add_76 = None
        mm_69 = torch.mm(view_248,t_77) ;  view_248 = t_77 = None
        view_249 = mm_69.view(1, 5, 512, 2304) ;  mm_69 = None
        view_250 = view_249.view(1, 5, 512, 16, 144) ;  view_249 = None
        permute_45 = view_250.permute(0, 3, 1, 2, 4) ;  view_250 = None
        split_tensor_34 = torch.split(permute_45,48,dim = -1) ;  permute_45 = None
        getitem_191 = split_tensor_34[0]
        getitem_192 = split_tensor_34[1]
        getitem_193 = split_tensor_34[2];  split_tensor_34 = None
        view_251 = arg133_1.view(1, 16, 1, 1, 48) ;  arg133_1 = None
        add_77 = torch.add(getitem_191,view_251) ;  getitem_191 = view_251 = None
        view_252 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_14 = torch.bitwise_not(view_252) ;  view_252 = None
        masked_fill_14 = view_245.masked_fill(bitwise_not_14,-10000) ;  view_245 = bitwise_not_14 = None
        permute_46 = masked_fill_14.permute(0, 4, 1, 2, 3) ;  masked_fill_14 = None
        view_253 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_15 = torch.bitwise_not(view_253) ;  view_253 = None
        masked_fill_15 = permute_46.masked_fill(bitwise_not_15,-10000) ;  permute_46 = bitwise_not_15 = None
        mul_53 = torch.mul(add_77,0.3799178428257963) ;  add_77 = None
        transpose_4 = torch.transpose(getitem_192,-2,-1) ;  getitem_192 = None
        mul_54 = torch.mul(transpose_4,0.3799178428257963) ;  transpose_4 = None
        expand_30 = mul_53.expand(1, 16, 5, 512, 48) ;  mul_53 = None
        clone_38 = torch.clone(expand_30,memory_format = torch.contiguous_format) ;  expand_30 = None
        _unsafe_view_37 = clone_38.view(80, 512, 48) ;  clone_38 = None
        expand_31 = mul_54.expand(1, 16, 5, 48, 512) ;  mul_54 = None
        clone_39 = torch.clone(expand_31,memory_format = torch.contiguous_format) ;  expand_31 = None
        _unsafe_view_38 = clone_39.view(80, 48, 512) ;  clone_39 = None
        bmm_14 = torch.bmm(_unsafe_view_37,_unsafe_view_38) ;  _unsafe_view_37 = _unsafe_view_38 = None
        view_254 = bmm_14.view(1, 16, 5, 512, 512) ;  bmm_14 = None
        add_78 = torch.add(view_254,masked_fill_15) ;  view_254 = masked_fill_15 = None
        _softmax_4 = torch.softmax(add_78, -1)
        expand_32 = _softmax_4.expand(1, 16, 5, 512, 512) ;  _softmax_4 = None
        view_255 = expand_32.view(80, 512, 512) ;  expand_32 = None
        expand_33 = getitem_193.expand(1, 16, 5, 512, 48) ;  getitem_193 = None
        clone_40 = torch.clone(expand_33,memory_format = torch.contiguous_format) ;  expand_33 = None
        _unsafe_view_39 = clone_40.view(80, 512, 48) ;  clone_40 = None
        bmm_15 = torch.bmm(view_255,_unsafe_view_39) ;  view_255 = _unsafe_view_39 = None
        view_256 = bmm_15.view(1, 16, 5, 512, 48) ;  bmm_15 = None
        permute_47 = view_256.permute(0, 2, 3, 1, 4) ;  view_256 = None
        clone_41 = torch.clone(permute_47,memory_format = torch.contiguous_format) ;  permute_47 = None
        _unsafe_view_40 = clone_41.view(1, 5, 512, 768) ;  clone_41 = None
        t_78 = arg146_1.t() ;  arg146_1 = None
        view_257 = _unsafe_view_40.view(2560, 768) ;  _unsafe_view_40 = None
        mm_70 = torch.mm(view_257,t_78) ;  view_257 = t_78 = None
        view_258 = mm_70.view(1, 5, 512, 768) ;  mm_70 = None
        view_259 = getitem_33.view(2560, 384) 
        t_79 = arg141_1.t() ;  arg141_1 = None
        addmm_8 = torch.addmm(arg142_1,view_259,t_79) ;  arg142_1 = view_259 = t_79 = None
        view_260 = addmm_8.view(1, 5, 512, 768) ;  addmm_8 = None
        sigmoid_14 = torch.sigmoid(view_260) ;  view_260 = None
        mul_55 = torch.mul(sigmoid_14,view_258) ;  sigmoid_14 = view_258 = None
        native_layer_norm_default_33 = (torch.nn.functional.layer_norm(add_74,[768],None,None,0.1),) 
        getitem_194 = native_layer_norm_default_33[0]
        t_80 = arg134_1.t() ;  arg134_1 = None
        view_261 = getitem_33.view(2560, 384) 
        mm_71 = torch.mm(view_261,t_80) ;  view_261 = t_80 = None
        view_262 = mm_71.view(1, 5, 512, 1536) ;  mm_71 = None
        split_tensor_35 = torch.split(view_262,768,dim = -1) ;  view_262 = None
        getitem_197 = split_tensor_35[0]
        getitem_198 = split_tensor_35[1];  split_tensor_35 = None
        add_79 = torch.add(getitem_197,1) ;  getitem_197 = None
        mul_56 = torch.mul(getitem_194,add_79) ;  getitem_194 = add_79 = None
        add_80 = torch.add(mul_56,getitem_198) ;  mul_56 = getitem_198 = None
        t_81 = arg135_1.t() ;  arg135_1 = None
        view_263 = add_80.view(2560, 768) ;  add_80 = None
        mm_72 = torch.mm(view_263,t_81) ;  view_263 = t_81 = None
        view_264 = mm_72.view(1, 5, 512, 3072) ;  mm_72 = None
        split_tensor_36 = torch.split(view_264,1536,dim = -1) ;  view_264 = None
        getitem_199 = split_tensor_36[0]
        getitem_200 = split_tensor_36[1];  split_tensor_36 = None
        silu_11 = torch.nn.functional.silu(getitem_199) ;  getitem_199 = None
        mul_57 = torch.mul(silu_11,getitem_200) ;  silu_11 = getitem_200 = None
        view_265 = getitem_33.view(2560, 384) 
        t_82 = arg137_1.t() ;  arg137_1 = None
        addmm_9 = torch.addmm(arg138_1,view_265,t_82) ;  arg138_1 = view_265 = t_82 = None
        view_266 = addmm_9.view(1, 5, 512, 768) ;  addmm_9 = None
        sigmoid_15 = torch.sigmoid(view_266) ;  view_266 = None
        t_83 = arg136_1.t() ;  arg136_1 = None
        view_267 = mul_57.view(2560, 1536) ;  mul_57 = None
        mm_73 = torch.mm(view_267,t_83) ;  view_267 = t_83 = None
        view_268 = mm_73.view(1, 5, 512, 768) ;  mm_73 = None
        mul_58 = torch.mul(sigmoid_15,view_268) ;  sigmoid_15 = view_268 = None
        add_81 = torch.add(mul_55,mul_58) ;  mul_55 = mul_58 = None
        add_82 = torch.add(add_74,add_81) ;  add_74 = add_81 = None
        native_layer_norm_default_34 = (torch.nn.functional.layer_norm(view_140,[256],arg157_1,arg158_1,1e-05),) ;  arg157_1 = arg158_1 = None
        getitem_201 = native_layer_norm_default_34[0]
        t_84 = arg159_1.t() ;  arg159_1 = None
        view_269 = getitem_201.view(262144, 256) ;  getitem_201 = None
        mm_74 = torch.mm(view_269,t_84) ;  view_269 = t_84 = None
        view_270 = mm_74.view(1, 1, 512, 512, 16) ;  mm_74 = None
        native_layer_norm_default_35 = (torch.nn.functional.layer_norm(add_82,[768],None,None,0.1),) 
        getitem_204 = native_layer_norm_default_35[0]
        t_85 = arg153_1.t() ;  arg153_1 = None
        view_271 = getitem_33.view(2560, 384) 
        mm_75 = torch.mm(view_271,t_85) ;  view_271 = t_85 = None
        view_272 = mm_75.view(1, 5, 512, 1536) ;  mm_75 = None
        split_tensor_37 = torch.split(view_272,768,dim = -1) ;  view_272 = None
        getitem_207 = split_tensor_37[0]
        getitem_208 = split_tensor_37[1];  split_tensor_37 = None
        add_83 = torch.add(getitem_207,1) ;  getitem_207 = None
        mul_59 = torch.mul(getitem_204,add_83) ;  getitem_204 = add_83 = None
        add_84 = torch.add(mul_59,getitem_208) ;  mul_59 = getitem_208 = None
        t_86 = arg154_1.t() ;  arg154_1 = None
        view_273 = add_84.view(2560, 768) ;  add_84 = None
        mm_76 = torch.mm(view_273,t_86) ;  view_273 = t_86 = None
        view_274 = mm_76.view(1, 5, 512, 2304) ;  mm_76 = None
        view_275 = view_274.view(1, 5, 512, 16, 144) ;  view_274 = None
        permute_48 = view_275.permute(0, 3, 1, 2, 4) ;  view_275 = None
        split_tensor_38 = torch.split(permute_48,48,dim = -1) ;  permute_48 = None
        getitem_209 = split_tensor_38[0]
        getitem_210 = split_tensor_38[1]
        getitem_211 = split_tensor_38[2];  split_tensor_38 = None
        view_276 = arg147_1.view(1, 16, 1, 1, 48) ;  arg147_1 = None
        add_85 = torch.add(getitem_209,view_276) ;  getitem_209 = view_276 = None
        view_277 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_16 = torch.bitwise_not(view_277) ;  view_277 = None
        masked_fill_16 = view_270.masked_fill(bitwise_not_16,-10000) ;  view_270 = bitwise_not_16 = None
        permute_49 = masked_fill_16.permute(0, 4, 1, 2, 3) ;  masked_fill_16 = None
        view_278 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_17 = torch.bitwise_not(view_278) ;  view_278 = None
        masked_fill_17 = permute_49.masked_fill(bitwise_not_17,-10000) ;  permute_49 = bitwise_not_17 = None
        mul_60 = torch.mul(add_85,0.3799178428257963) ;  add_85 = None
        transpose_5 = torch.transpose(getitem_210,-2,-1) ;  getitem_210 = None
        mul_61 = torch.mul(transpose_5,0.3799178428257963) ;  transpose_5 = None
        expand_34 = mul_60.expand(1, 16, 5, 512, 48) ;  mul_60 = None
        clone_42 = torch.clone(expand_34,memory_format = torch.contiguous_format) ;  expand_34 = None
        _unsafe_view_41 = clone_42.view(80, 512, 48) ;  clone_42 = None
        expand_35 = mul_61.expand(1, 16, 5, 48, 512) ;  mul_61 = None
        clone_43 = torch.clone(expand_35,memory_format = torch.contiguous_format) ;  expand_35 = None
        _unsafe_view_42 = clone_43.view(80, 48, 512) ;  clone_43 = None
        bmm_16 = torch.bmm(_unsafe_view_41,_unsafe_view_42) ;  _unsafe_view_41 = _unsafe_view_42 = None
        view_279 = bmm_16.view(1, 16, 5, 512, 512) ;  bmm_16 = None
        add_86 = torch.add(view_279,masked_fill_17) ;  view_279 = masked_fill_17 = None
        _softmax_5 = torch.softmax(add_86, -1)
        expand_36 = _softmax_5.expand(1, 16, 5, 512, 512) ;  _softmax_5 = None
        view_280 = expand_36.view(80, 512, 512) ;  expand_36 = None
        expand_37 = getitem_211.expand(1, 16, 5, 512, 48) ;  getitem_211 = None
        clone_44 = torch.clone(expand_37,memory_format = torch.contiguous_format) ;  expand_37 = None
        _unsafe_view_43 = clone_44.view(80, 512, 48) ;  clone_44 = None
        bmm_17 = torch.bmm(view_280,_unsafe_view_43) ;  view_280 = _unsafe_view_43 = None
        view_281 = bmm_17.view(1, 16, 5, 512, 48) ;  bmm_17 = None
        permute_50 = view_281.permute(0, 2, 3, 1, 4) ;  view_281 = None
        clone_45 = torch.clone(permute_50,memory_format = torch.contiguous_format) ;  permute_50 = None
        _unsafe_view_44 = clone_45.view(1, 5, 512, 768) ;  clone_45 = None
        t_87 = arg160_1.t() ;  arg160_1 = None
        view_282 = _unsafe_view_44.view(2560, 768) ;  _unsafe_view_44 = None
        mm_77 = torch.mm(view_282,t_87) ;  view_282 = t_87 = None
        view_283 = mm_77.view(1, 5, 512, 768) ;  mm_77 = None
        view_284 = getitem_33.view(2560, 384) 
        t_88 = arg155_1.t() ;  arg155_1 = None
        addmm_10 = torch.addmm(arg156_1,view_284,t_88) ;  arg156_1 = view_284 = t_88 = None
        view_285 = addmm_10.view(1, 5, 512, 768) ;  addmm_10 = None
        sigmoid_16 = torch.sigmoid(view_285) ;  view_285 = None
        mul_62 = torch.mul(sigmoid_16,view_283) ;  sigmoid_16 = view_283 = None
        native_layer_norm_default_36 = (torch.nn.functional.layer_norm(add_82,[768],None,None,0.1),) 
        getitem_212 = native_layer_norm_default_36[0]
        t_89 = arg148_1.t() ;  arg148_1 = None
        view_286 = getitem_33.view(2560, 384) 
        mm_78 = torch.mm(view_286,t_89) ;  view_286 = t_89 = None
        view_287 = mm_78.view(1, 5, 512, 1536) ;  mm_78 = None
        split_tensor_39 = torch.split(view_287,768,dim = -1) ;  view_287 = None
        getitem_215 = split_tensor_39[0]
        getitem_216 = split_tensor_39[1];  split_tensor_39 = None
        add_87 = torch.add(getitem_215,1) ;  getitem_215 = None
        mul_63 = torch.mul(getitem_212,add_87) ;  getitem_212 = add_87 = None
        add_88 = torch.add(mul_63,getitem_216) ;  mul_63 = getitem_216 = None
        t_90 = arg149_1.t() ;  arg149_1 = None
        view_288 = add_88.view(2560, 768) ;  add_88 = None
        mm_79 = torch.mm(view_288,t_90) ;  view_288 = t_90 = None
        view_289 = mm_79.view(1, 5, 512, 3072) ;  mm_79 = None
        split_tensor_40 = torch.split(view_289,1536,dim = -1) ;  view_289 = None
        getitem_217 = split_tensor_40[0]
        getitem_218 = split_tensor_40[1];  split_tensor_40 = None
        silu_12 = torch.nn.functional.silu(getitem_217) ;  getitem_217 = None
        mul_64 = torch.mul(silu_12,getitem_218) ;  silu_12 = getitem_218 = None
        view_290 = getitem_33.view(2560, 384) 
        t_91 = arg151_1.t() ;  arg151_1 = None
        addmm_11 = torch.addmm(arg152_1,view_290,t_91) ;  arg152_1 = view_290 = t_91 = None
        view_291 = addmm_11.view(1, 5, 512, 768) ;  addmm_11 = None
        sigmoid_17 = torch.sigmoid(view_291) ;  view_291 = None
        t_92 = arg150_1.t() ;  arg150_1 = None
        view_292 = mul_64.view(2560, 1536) ;  mul_64 = None
        mm_80 = torch.mm(view_292,t_92) ;  view_292 = t_92 = None
        view_293 = mm_80.view(1, 5, 512, 768) ;  mm_80 = None
        mul_65 = torch.mul(sigmoid_17,view_293) ;  sigmoid_17 = view_293 = None
        add_89 = torch.add(mul_62,mul_65) ;  mul_62 = mul_65 = None
        add_90 = torch.add(add_82,add_89) ;  add_82 = add_89 = None
        native_layer_norm_default_37 = (torch.nn.functional.layer_norm(view_140,[256],arg171_1,arg172_1,1e-05),) ;  arg171_1 = arg172_1 = None
        getitem_219 = native_layer_norm_default_37[0]
        t_93 = arg173_1.t() ;  arg173_1 = None
        view_294 = getitem_219.view(262144, 256) ;  getitem_219 = None
        mm_81 = torch.mm(view_294,t_93) ;  view_294 = t_93 = None
        view_295 = mm_81.view(1, 1, 512, 512, 16) ;  mm_81 = None
        native_layer_norm_default_38 = (torch.nn.functional.layer_norm(add_90,[768],None,None,0.1),) 
        getitem_222 = native_layer_norm_default_38[0]
        t_94 = arg167_1.t() ;  arg167_1 = None
        view_296 = getitem_33.view(2560, 384) 
        mm_82 = torch.mm(view_296,t_94) ;  view_296 = t_94 = None
        view_297 = mm_82.view(1, 5, 512, 1536) ;  mm_82 = None
        split_tensor_41 = torch.split(view_297,768,dim = -1) ;  view_297 = None
        getitem_225 = split_tensor_41[0]
        getitem_226 = split_tensor_41[1];  split_tensor_41 = None
        add_91 = torch.add(getitem_225,1) ;  getitem_225 = None
        mul_66 = torch.mul(getitem_222,add_91) ;  getitem_222 = add_91 = None
        add_92 = torch.add(mul_66,getitem_226) ;  mul_66 = getitem_226 = None
        t_95 = arg168_1.t() ;  arg168_1 = None
        view_298 = add_92.view(2560, 768) ;  add_92 = None
        mm_83 = torch.mm(view_298,t_95) ;  view_298 = t_95 = None
        view_299 = mm_83.view(1, 5, 512, 2304) ;  mm_83 = None
        view_300 = view_299.view(1, 5, 512, 16, 144) ;  view_299 = None
        permute_51 = view_300.permute(0, 3, 1, 2, 4) ;  view_300 = None
        split_tensor_42 = torch.split(permute_51,48,dim = -1) ;  permute_51 = None
        getitem_227 = split_tensor_42[0]
        getitem_228 = split_tensor_42[1]
        getitem_229 = split_tensor_42[2];  split_tensor_42 = None
        view_301 = arg161_1.view(1, 16, 1, 1, 48) ;  arg161_1 = None
        add_93 = torch.add(getitem_227,view_301) ;  getitem_227 = view_301 = None
        view_302 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_18 = torch.bitwise_not(view_302) ;  view_302 = None
        masked_fill_18 = view_295.masked_fill(bitwise_not_18,-10000) ;  view_295 = bitwise_not_18 = None
        permute_52 = masked_fill_18.permute(0, 4, 1, 2, 3) ;  masked_fill_18 = None
        view_303 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_19 = torch.bitwise_not(view_303) ;  view_303 = None
        masked_fill_19 = permute_52.masked_fill(bitwise_not_19,-10000) ;  permute_52 = bitwise_not_19 = None
        mul_67 = torch.mul(add_93,0.3799178428257963) ;  add_93 = None
        transpose_6 = torch.transpose(getitem_228,-2,-1) ;  getitem_228 = None
        mul_68 = torch.mul(transpose_6,0.3799178428257963) ;  transpose_6 = None
        expand_38 = mul_67.expand(1, 16, 5, 512, 48) ;  mul_67 = None
        clone_46 = torch.clone(expand_38,memory_format = torch.contiguous_format) ;  expand_38 = None
        _unsafe_view_45 = clone_46.view(80, 512, 48) ;  clone_46 = None
        expand_39 = mul_68.expand(1, 16, 5, 48, 512) ;  mul_68 = None
        clone_47 = torch.clone(expand_39,memory_format = torch.contiguous_format) ;  expand_39 = None
        _unsafe_view_46 = clone_47.view(80, 48, 512) ;  clone_47 = None
        bmm_18 = torch.bmm(_unsafe_view_45,_unsafe_view_46) ;  _unsafe_view_45 = _unsafe_view_46 = None
        view_304 = bmm_18.view(1, 16, 5, 512, 512) ;  bmm_18 = None
        add_94 = torch.add(view_304,masked_fill_19) ;  view_304 = masked_fill_19 = None
        _softmax_6 = torch.softmax(add_94, -1)
        expand_40 = _softmax_6.expand(1, 16, 5, 512, 512) ;  _softmax_6 = None
        view_305 = expand_40.view(80, 512, 512) ;  expand_40 = None
        expand_41 = getitem_229.expand(1, 16, 5, 512, 48) ;  getitem_229 = None
        clone_48 = torch.clone(expand_41,memory_format = torch.contiguous_format) ;  expand_41 = None
        _unsafe_view_47 = clone_48.view(80, 512, 48) ;  clone_48 = None
        bmm_19 = torch.bmm(view_305,_unsafe_view_47) ;  view_305 = _unsafe_view_47 = None
        view_306 = bmm_19.view(1, 16, 5, 512, 48) ;  bmm_19 = None
        permute_53 = view_306.permute(0, 2, 3, 1, 4) ;  view_306 = None
        clone_49 = torch.clone(permute_53,memory_format = torch.contiguous_format) ;  permute_53 = None
        _unsafe_view_48 = clone_49.view(1, 5, 512, 768) ;  clone_49 = None
        t_96 = arg174_1.t() ;  arg174_1 = None
        view_307 = _unsafe_view_48.view(2560, 768) ;  _unsafe_view_48 = None
        mm_84 = torch.mm(view_307,t_96) ;  view_307 = t_96 = None
        view_308 = mm_84.view(1, 5, 512, 768) ;  mm_84 = None
        view_309 = getitem_33.view(2560, 384) 
        t_97 = arg169_1.t() ;  arg169_1 = None
        addmm_12 = torch.addmm(arg170_1,view_309,t_97) ;  arg170_1 = view_309 = t_97 = None
        view_310 = addmm_12.view(1, 5, 512, 768) ;  addmm_12 = None
        sigmoid_18 = torch.sigmoid(view_310) ;  view_310 = None
        mul_69 = torch.mul(sigmoid_18,view_308) ;  sigmoid_18 = view_308 = None
        native_layer_norm_default_39 = (torch.nn.functional.layer_norm(add_90,[768],None,None,0.1),) 
        getitem_230 = native_layer_norm_default_39[0]
        t_98 = arg162_1.t() ;  arg162_1 = None
        view_311 = getitem_33.view(2560, 384) 
        mm_85 = torch.mm(view_311,t_98) ;  view_311 = t_98 = None
        view_312 = mm_85.view(1, 5, 512, 1536) ;  mm_85 = None
        split_tensor_43 = torch.split(view_312,768,dim = -1) ;  view_312 = None
        getitem_233 = split_tensor_43[0]
        getitem_234 = split_tensor_43[1];  split_tensor_43 = None
        add_95 = torch.add(getitem_233,1) ;  getitem_233 = None
        mul_70 = torch.mul(getitem_230,add_95) ;  getitem_230 = add_95 = None
        add_96 = torch.add(mul_70,getitem_234) ;  mul_70 = getitem_234 = None
        t_99 = arg163_1.t() ;  arg163_1 = None
        view_313 = add_96.view(2560, 768) ;  add_96 = None
        mm_86 = torch.mm(view_313,t_99) ;  view_313 = t_99 = None
        view_314 = mm_86.view(1, 5, 512, 3072) ;  mm_86 = None
        split_tensor_44 = torch.split(view_314,1536,dim = -1) ;  view_314 = None
        getitem_235 = split_tensor_44[0]
        getitem_236 = split_tensor_44[1];  split_tensor_44 = None
        silu_13 = torch.nn.functional.silu(getitem_235) ;  getitem_235 = None
        mul_71 = torch.mul(silu_13,getitem_236) ;  silu_13 = getitem_236 = None
        view_315 = getitem_33.view(2560, 384) 
        t_100 = arg165_1.t() ;  arg165_1 = None
        addmm_13 = torch.addmm(arg166_1,view_315,t_100) ;  arg166_1 = view_315 = t_100 = None
        view_316 = addmm_13.view(1, 5, 512, 768) ;  addmm_13 = None
        sigmoid_19 = torch.sigmoid(view_316) ;  view_316 = None
        t_101 = arg164_1.t() ;  arg164_1 = None
        view_317 = mul_71.view(2560, 1536) ;  mul_71 = None
        mm_87 = torch.mm(view_317,t_101) ;  view_317 = t_101 = None
        view_318 = mm_87.view(1, 5, 512, 768) ;  mm_87 = None
        mul_72 = torch.mul(sigmoid_19,view_318) ;  sigmoid_19 = view_318 = None
        add_97 = torch.add(mul_69,mul_72) ;  mul_69 = mul_72 = None
        add_98 = torch.add(add_90,add_97) ;  add_90 = add_97 = None
        native_layer_norm_default_40 = (torch.nn.functional.layer_norm(view_140,[256],arg185_1,arg186_1,1e-05),) ;  arg185_1 = arg186_1 = None
        getitem_237 = native_layer_norm_default_40[0]
        t_102 = arg187_1.t() ;  arg187_1 = None
        view_319 = getitem_237.view(262144, 256) ;  getitem_237 = None
        mm_88 = torch.mm(view_319,t_102) ;  view_319 = t_102 = None
        view_320 = mm_88.view(1, 1, 512, 512, 16) ;  mm_88 = None
        native_layer_norm_default_41 = (torch.nn.functional.layer_norm(add_98,[768],None,None,0.1),) 
        getitem_240 = native_layer_norm_default_41[0]
        t_103 = arg181_1.t() ;  arg181_1 = None
        view_321 = getitem_33.view(2560, 384) 
        mm_89 = torch.mm(view_321,t_103) ;  view_321 = t_103 = None
        view_322 = mm_89.view(1, 5, 512, 1536) ;  mm_89 = None
        split_tensor_45 = torch.split(view_322,768,dim = -1) ;  view_322 = None
        getitem_243 = split_tensor_45[0]
        getitem_244 = split_tensor_45[1];  split_tensor_45 = None
        add_99 = torch.add(getitem_243,1) ;  getitem_243 = None
        mul_73 = torch.mul(getitem_240,add_99) ;  getitem_240 = add_99 = None
        add_100 = torch.add(mul_73,getitem_244) ;  mul_73 = getitem_244 = None
        t_104 = arg182_1.t() ;  arg182_1 = None
        view_323 = add_100.view(2560, 768) ;  add_100 = None
        mm_90 = torch.mm(view_323,t_104) ;  view_323 = t_104 = None
        view_324 = mm_90.view(1, 5, 512, 2304) ;  mm_90 = None
        view_325 = view_324.view(1, 5, 512, 16, 144) ;  view_324 = None
        permute_54 = view_325.permute(0, 3, 1, 2, 4) ;  view_325 = None
        split_tensor_46 = torch.split(permute_54,48,dim = -1) ;  permute_54 = None
        getitem_245 = split_tensor_46[0]
        getitem_246 = split_tensor_46[1]
        getitem_247 = split_tensor_46[2];  split_tensor_46 = None
        view_326 = arg175_1.view(1, 16, 1, 1, 48) ;  arg175_1 = None
        add_101 = torch.add(getitem_245,view_326) ;  getitem_245 = view_326 = None
        view_327 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_20 = torch.bitwise_not(view_327) ;  view_327 = None
        masked_fill_20 = view_320.masked_fill(bitwise_not_20,-10000) ;  view_320 = bitwise_not_20 = None
        permute_55 = masked_fill_20.permute(0, 4, 1, 2, 3) ;  masked_fill_20 = None
        view_328 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_21 = torch.bitwise_not(view_328) ;  view_328 = None
        masked_fill_21 = permute_55.masked_fill(bitwise_not_21,-10000) ;  permute_55 = bitwise_not_21 = None
        mul_74 = torch.mul(add_101,0.3799178428257963) ;  add_101 = None
        transpose_7 = torch.transpose(getitem_246,-2,-1) ;  getitem_246 = None
        mul_75 = torch.mul(transpose_7,0.3799178428257963) ;  transpose_7 = None
        expand_42 = mul_74.expand(1, 16, 5, 512, 48) ;  mul_74 = None
        clone_50 = torch.clone(expand_42,memory_format = torch.contiguous_format) ;  expand_42 = None
        _unsafe_view_49 = clone_50.view(80, 512, 48) ;  clone_50 = None
        expand_43 = mul_75.expand(1, 16, 5, 48, 512) ;  mul_75 = None
        clone_51 = torch.clone(expand_43,memory_format = torch.contiguous_format) ;  expand_43 = None
        _unsafe_view_50 = clone_51.view(80, 48, 512) ;  clone_51 = None
        bmm_20 = torch.bmm(_unsafe_view_49,_unsafe_view_50) ;  _unsafe_view_49 = _unsafe_view_50 = None
        view_329 = bmm_20.view(1, 16, 5, 512, 512) ;  bmm_20 = None
        add_102 = torch.add(view_329,masked_fill_21) ;  view_329 = masked_fill_21 = None
        _softmax_7 = torch.softmax(add_102, -1)
        expand_44 = _softmax_7.expand(1, 16, 5, 512, 512) ;  _softmax_7 = None
        view_330 = expand_44.view(80, 512, 512) ;  expand_44 = None
        expand_45 = getitem_247.expand(1, 16, 5, 512, 48) ;  getitem_247 = None
        clone_52 = torch.clone(expand_45,memory_format = torch.contiguous_format) ;  expand_45 = None
        _unsafe_view_51 = clone_52.view(80, 512, 48) ;  clone_52 = None
        bmm_21 = torch.bmm(view_330,_unsafe_view_51) ;  view_330 = _unsafe_view_51 = None
        view_331 = bmm_21.view(1, 16, 5, 512, 48) ;  bmm_21 = None
        permute_56 = view_331.permute(0, 2, 3, 1, 4) ;  view_331 = None
        clone_53 = torch.clone(permute_56,memory_format = torch.contiguous_format) ;  permute_56 = None
        _unsafe_view_52 = clone_53.view(1, 5, 512, 768) ;  clone_53 = None
        t_105 = arg188_1.t() ;  arg188_1 = None
        view_332 = _unsafe_view_52.view(2560, 768) ;  _unsafe_view_52 = None
        mm_91 = torch.mm(view_332,t_105) ;  view_332 = t_105 = None
        view_333 = mm_91.view(1, 5, 512, 768) ;  mm_91 = None
        view_334 = getitem_33.view(2560, 384) 
        t_106 = arg183_1.t() ;  arg183_1 = None
        addmm_14 = torch.addmm(arg184_1,view_334,t_106) ;  arg184_1 = view_334 = t_106 = None
        view_335 = addmm_14.view(1, 5, 512, 768) ;  addmm_14 = None
        sigmoid_20 = torch.sigmoid(view_335) ;  view_335 = None
        mul_76 = torch.mul(sigmoid_20,view_333) ;  sigmoid_20 = view_333 = None
        native_layer_norm_default_42 = (torch.nn.functional.layer_norm(add_98,[768],None,None,0.1),) 
        getitem_248 = native_layer_norm_default_42[0]
        t_107 = arg176_1.t() ;  arg176_1 = None
        view_336 = getitem_33.view(2560, 384) 
        mm_92 = torch.mm(view_336,t_107) ;  view_336 = t_107 = None
        view_337 = mm_92.view(1, 5, 512, 1536) ;  mm_92 = None
        split_tensor_47 = torch.split(view_337,768,dim = -1) ;  view_337 = None
        getitem_251 = split_tensor_47[0]
        getitem_252 = split_tensor_47[1];  split_tensor_47 = None
        add_103 = torch.add(getitem_251,1) ;  getitem_251 = None
        mul_77 = torch.mul(getitem_248,add_103) ;  getitem_248 = add_103 = None
        add_104 = torch.add(mul_77,getitem_252) ;  mul_77 = getitem_252 = None
        t_108 = arg177_1.t() ;  arg177_1 = None
        view_338 = add_104.view(2560, 768) ;  add_104 = None
        mm_93 = torch.mm(view_338,t_108) ;  view_338 = t_108 = None
        view_339 = mm_93.view(1, 5, 512, 3072) ;  mm_93 = None
        split_tensor_48 = torch.split(view_339,1536,dim = -1) ;  view_339 = None
        getitem_253 = split_tensor_48[0]
        getitem_254 = split_tensor_48[1];  split_tensor_48 = None
        silu_14 = torch.nn.functional.silu(getitem_253) ;  getitem_253 = None
        mul_78 = torch.mul(silu_14,getitem_254) ;  silu_14 = getitem_254 = None
        view_340 = getitem_33.view(2560, 384) 
        t_109 = arg179_1.t() ;  arg179_1 = None
        addmm_15 = torch.addmm(arg180_1,view_340,t_109) ;  arg180_1 = view_340 = t_109 = None
        view_341 = addmm_15.view(1, 5, 512, 768) ;  addmm_15 = None
        sigmoid_21 = torch.sigmoid(view_341) ;  view_341 = None
        t_110 = arg178_1.t() ;  arg178_1 = None
        view_342 = mul_78.view(2560, 1536) ;  mul_78 = None
        mm_94 = torch.mm(view_342,t_110) ;  view_342 = t_110 = None
        view_343 = mm_94.view(1, 5, 512, 768) ;  mm_94 = None
        mul_79 = torch.mul(sigmoid_21,view_343) ;  sigmoid_21 = view_343 = None
        add_105 = torch.add(mul_76,mul_79) ;  mul_76 = mul_79 = None
        add_106 = torch.add(add_98,add_105) ;  add_98 = add_105 = None
        native_layer_norm_default_43 = (torch.nn.functional.layer_norm(view_140,[256],arg199_1,arg200_1,1e-05),) ;  arg199_1 = arg200_1 = None
        getitem_255 = native_layer_norm_default_43[0]
        t_111 = arg201_1.t() ;  arg201_1 = None
        view_344 = getitem_255.view(262144, 256) ;  getitem_255 = None
        mm_95 = torch.mm(view_344,t_111) ;  view_344 = t_111 = None
        view_345 = mm_95.view(1, 1, 512, 512, 16) ;  mm_95 = None
        native_layer_norm_default_44 = (torch.nn.functional.layer_norm(add_106,[768],None,None,0.1),) 
        getitem_258 = native_layer_norm_default_44[0]
        t_112 = arg195_1.t() ;  arg195_1 = None
        view_346 = getitem_33.view(2560, 384) 
        mm_96 = torch.mm(view_346,t_112) ;  view_346 = t_112 = None
        view_347 = mm_96.view(1, 5, 512, 1536) ;  mm_96 = None
        split_tensor_49 = torch.split(view_347,768,dim = -1) ;  view_347 = None
        getitem_261 = split_tensor_49[0]
        getitem_262 = split_tensor_49[1];  split_tensor_49 = None
        add_107 = torch.add(getitem_261,1) ;  getitem_261 = None
        mul_80 = torch.mul(getitem_258,add_107) ;  getitem_258 = add_107 = None
        add_108 = torch.add(mul_80,getitem_262) ;  mul_80 = getitem_262 = None
        t_113 = arg196_1.t() ;  arg196_1 = None
        view_348 = add_108.view(2560, 768) ;  add_108 = None
        mm_97 = torch.mm(view_348,t_113) ;  view_348 = t_113 = None
        view_349 = mm_97.view(1, 5, 512, 2304) ;  mm_97 = None
        view_350 = view_349.view(1, 5, 512, 16, 144) ;  view_349 = None
        permute_57 = view_350.permute(0, 3, 1, 2, 4) ;  view_350 = None
        split_tensor_50 = torch.split(permute_57,48,dim = -1) ;  permute_57 = None
        getitem_263 = split_tensor_50[0]
        getitem_264 = split_tensor_50[1]
        getitem_265 = split_tensor_50[2];  split_tensor_50 = None
        view_351 = arg189_1.view(1, 16, 1, 1, 48) ;  arg189_1 = None
        add_109 = torch.add(getitem_263,view_351) ;  getitem_263 = view_351 = None
        view_352 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_22 = torch.bitwise_not(view_352) ;  view_352 = None
        masked_fill_22 = view_345.masked_fill(bitwise_not_22,-10000) ;  view_345 = bitwise_not_22 = None
        permute_58 = masked_fill_22.permute(0, 4, 1, 2, 3) ;  masked_fill_22 = None
        view_353 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_23 = torch.bitwise_not(view_353) ;  view_353 = None
        masked_fill_23 = permute_58.masked_fill(bitwise_not_23,-10000) ;  permute_58 = bitwise_not_23 = None
        mul_81 = torch.mul(add_109,0.3799178428257963) ;  add_109 = None
        transpose_8 = torch.transpose(getitem_264,-2,-1) ;  getitem_264 = None
        mul_82 = torch.mul(transpose_8,0.3799178428257963) ;  transpose_8 = None
        expand_46 = mul_81.expand(1, 16, 5, 512, 48) ;  mul_81 = None
        clone_54 = torch.clone(expand_46,memory_format = torch.contiguous_format) ;  expand_46 = None
        _unsafe_view_53 = clone_54.view(80, 512, 48) ;  clone_54 = None
        expand_47 = mul_82.expand(1, 16, 5, 48, 512) ;  mul_82 = None
        clone_55 = torch.clone(expand_47,memory_format = torch.contiguous_format) ;  expand_47 = None
        _unsafe_view_54 = clone_55.view(80, 48, 512) ;  clone_55 = None
        bmm_22 = torch.bmm(_unsafe_view_53,_unsafe_view_54) ;  _unsafe_view_53 = _unsafe_view_54 = None
        view_354 = bmm_22.view(1, 16, 5, 512, 512) ;  bmm_22 = None
        add_110 = torch.add(view_354,masked_fill_23) ;  view_354 = masked_fill_23 = None
        _softmax_8 = torch.softmax(add_110, -1)
        expand_48 = _softmax_8.expand(1, 16, 5, 512, 512) ;  _softmax_8 = None
        view_355 = expand_48.view(80, 512, 512) ;  expand_48 = None
        expand_49 = getitem_265.expand(1, 16, 5, 512, 48) ;  getitem_265 = None
        clone_56 = torch.clone(expand_49,memory_format = torch.contiguous_format) ;  expand_49 = None
        _unsafe_view_55 = clone_56.view(80, 512, 48) ;  clone_56 = None
        bmm_23 = torch.bmm(view_355,_unsafe_view_55) ;  view_355 = _unsafe_view_55 = None
        view_356 = bmm_23.view(1, 16, 5, 512, 48) ;  bmm_23 = None
        permute_59 = view_356.permute(0, 2, 3, 1, 4) ;  view_356 = None
        clone_57 = torch.clone(permute_59,memory_format = torch.contiguous_format) ;  permute_59 = None
        _unsafe_view_56 = clone_57.view(1, 5, 512, 768) ;  clone_57 = None
        t_114 = arg202_1.t() ;  arg202_1 = None
        view_357 = _unsafe_view_56.view(2560, 768) ;  _unsafe_view_56 = None
        mm_98 = torch.mm(view_357,t_114) ;  view_357 = t_114 = None
        view_358 = mm_98.view(1, 5, 512, 768) ;  mm_98 = None
        view_359 = getitem_33.view(2560, 384) 
        t_115 = arg197_1.t() ;  arg197_1 = None
        addmm_16 = torch.addmm(arg198_1,view_359,t_115) ;  arg198_1 = view_359 = t_115 = None
        view_360 = addmm_16.view(1, 5, 512, 768) ;  addmm_16 = None
        sigmoid_22 = torch.sigmoid(view_360) ;  view_360 = None
        mul_83 = torch.mul(sigmoid_22,view_358) ;  sigmoid_22 = view_358 = None
        native_layer_norm_default_45 = (torch.nn.functional.layer_norm(add_106,[768],None,None,0.1),) 
        getitem_266 = native_layer_norm_default_45[0]
        t_116 = arg190_1.t() ;  arg190_1 = None
        view_361 = getitem_33.view(2560, 384) 
        mm_99 = torch.mm(view_361,t_116) ;  view_361 = t_116 = None
        view_362 = mm_99.view(1, 5, 512, 1536) ;  mm_99 = None
        split_tensor_51 = torch.split(view_362,768,dim = -1) ;  view_362 = None
        getitem_269 = split_tensor_51[0]
        getitem_270 = split_tensor_51[1];  split_tensor_51 = None
        add_111 = torch.add(getitem_269,1) ;  getitem_269 = None
        mul_84 = torch.mul(getitem_266,add_111) ;  getitem_266 = add_111 = None
        add_112 = torch.add(mul_84,getitem_270) ;  mul_84 = getitem_270 = None
        t_117 = arg191_1.t() ;  arg191_1 = None
        view_363 = add_112.view(2560, 768) ;  add_112 = None
        mm_100 = torch.mm(view_363,t_117) ;  view_363 = t_117 = None
        view_364 = mm_100.view(1, 5, 512, 3072) ;  mm_100 = None
        split_tensor_52 = torch.split(view_364,1536,dim = -1) ;  view_364 = None
        getitem_271 = split_tensor_52[0]
        getitem_272 = split_tensor_52[1];  split_tensor_52 = None
        silu_15 = torch.nn.functional.silu(getitem_271) ;  getitem_271 = None
        mul_85 = torch.mul(silu_15,getitem_272) ;  silu_15 = getitem_272 = None
        view_365 = getitem_33.view(2560, 384) 
        t_118 = arg193_1.t() ;  arg193_1 = None
        addmm_17 = torch.addmm(arg194_1,view_365,t_118) ;  arg194_1 = view_365 = t_118 = None
        view_366 = addmm_17.view(1, 5, 512, 768) ;  addmm_17 = None
        sigmoid_23 = torch.sigmoid(view_366) ;  view_366 = None
        t_119 = arg192_1.t() ;  arg192_1 = None
        view_367 = mul_85.view(2560, 1536) ;  mul_85 = None
        mm_101 = torch.mm(view_367,t_119) ;  view_367 = t_119 = None
        view_368 = mm_101.view(1, 5, 512, 768) ;  mm_101 = None
        mul_86 = torch.mul(sigmoid_23,view_368) ;  sigmoid_23 = view_368 = None
        add_113 = torch.add(mul_83,mul_86) ;  mul_83 = mul_86 = None
        add_114 = torch.add(add_106,add_113) ;  add_106 = add_113 = None
        native_layer_norm_default_46 = (torch.nn.functional.layer_norm(view_140,[256],arg213_1,arg214_1,1e-05),) ;  arg213_1 = arg214_1 = None
        getitem_273 = native_layer_norm_default_46[0]
        t_120 = arg215_1.t() ;  arg215_1 = None
        view_369 = getitem_273.view(262144, 256) ;  getitem_273 = None
        mm_102 = torch.mm(view_369,t_120) ;  view_369 = t_120 = None
        view_370 = mm_102.view(1, 1, 512, 512, 16) ;  mm_102 = None
        native_layer_norm_default_47 = (torch.nn.functional.layer_norm(add_114,[768],None,None,0.1),) 
        getitem_276 = native_layer_norm_default_47[0]
        t_121 = arg209_1.t() ;  arg209_1 = None
        view_371 = getitem_33.view(2560, 384) 
        mm_103 = torch.mm(view_371,t_121) ;  view_371 = t_121 = None
        view_372 = mm_103.view(1, 5, 512, 1536) ;  mm_103 = None
        split_tensor_53 = torch.split(view_372,768,dim = -1) ;  view_372 = None
        getitem_279 = split_tensor_53[0]
        getitem_280 = split_tensor_53[1];  split_tensor_53 = None
        add_115 = torch.add(getitem_279,1) ;  getitem_279 = None
        mul_87 = torch.mul(getitem_276,add_115) ;  getitem_276 = add_115 = None
        add_116 = torch.add(mul_87,getitem_280) ;  mul_87 = getitem_280 = None
        t_122 = arg210_1.t() ;  arg210_1 = None
        view_373 = add_116.view(2560, 768) ;  add_116 = None
        mm_104 = torch.mm(view_373,t_122) ;  view_373 = t_122 = None
        view_374 = mm_104.view(1, 5, 512, 2304) ;  mm_104 = None
        view_375 = view_374.view(1, 5, 512, 16, 144) ;  view_374 = None
        permute_60 = view_375.permute(0, 3, 1, 2, 4) ;  view_375 = None
        split_tensor_54 = torch.split(permute_60,48,dim = -1) ;  permute_60 = None
        getitem_281 = split_tensor_54[0]
        getitem_282 = split_tensor_54[1]
        getitem_283 = split_tensor_54[2];  split_tensor_54 = None
        view_376 = arg203_1.view(1, 16, 1, 1, 48) ;  arg203_1 = None
        add_117 = torch.add(getitem_281,view_376) ;  getitem_281 = view_376 = None
        view_377 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_24 = torch.bitwise_not(view_377) ;  view_377 = None
        masked_fill_24 = view_370.masked_fill(bitwise_not_24,-10000) ;  view_370 = bitwise_not_24 = None
        permute_61 = masked_fill_24.permute(0, 4, 1, 2, 3) ;  masked_fill_24 = None
        view_378 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_25 = torch.bitwise_not(view_378) ;  view_378 = None
        masked_fill_25 = permute_61.masked_fill(bitwise_not_25,-10000) ;  permute_61 = bitwise_not_25 = None
        mul_88 = torch.mul(add_117,0.3799178428257963) ;  add_117 = None
        transpose_9 = torch.transpose(getitem_282,-2,-1) ;  getitem_282 = None
        mul_89 = torch.mul(transpose_9,0.3799178428257963) ;  transpose_9 = None
        expand_50 = mul_88.expand(1, 16, 5, 512, 48) ;  mul_88 = None
        clone_58 = torch.clone(expand_50,memory_format = torch.contiguous_format) ;  expand_50 = None
        _unsafe_view_57 = clone_58.view(80, 512, 48) ;  clone_58 = None
        expand_51 = mul_89.expand(1, 16, 5, 48, 512) ;  mul_89 = None
        clone_59 = torch.clone(expand_51,memory_format = torch.contiguous_format) ;  expand_51 = None
        _unsafe_view_58 = clone_59.view(80, 48, 512) ;  clone_59 = None
        bmm_24 = torch.bmm(_unsafe_view_57,_unsafe_view_58) ;  _unsafe_view_57 = _unsafe_view_58 = None
        view_379 = bmm_24.view(1, 16, 5, 512, 512) ;  bmm_24 = None
        add_118 = torch.add(view_379,masked_fill_25) ;  view_379 = masked_fill_25 = None
        _softmax_9 = torch.softmax(add_118, -1)
        expand_52 = _softmax_9.expand(1, 16, 5, 512, 512) ;  _softmax_9 = None
        view_380 = expand_52.view(80, 512, 512) ;  expand_52 = None
        expand_53 = getitem_283.expand(1, 16, 5, 512, 48) ;  getitem_283 = None
        clone_60 = torch.clone(expand_53,memory_format = torch.contiguous_format) ;  expand_53 = None
        _unsafe_view_59 = clone_60.view(80, 512, 48) ;  clone_60 = None
        bmm_25 = torch.bmm(view_380,_unsafe_view_59) ;  view_380 = _unsafe_view_59 = None
        view_381 = bmm_25.view(1, 16, 5, 512, 48) ;  bmm_25 = None
        permute_62 = view_381.permute(0, 2, 3, 1, 4) ;  view_381 = None
        clone_61 = torch.clone(permute_62,memory_format = torch.contiguous_format) ;  permute_62 = None
        _unsafe_view_60 = clone_61.view(1, 5, 512, 768) ;  clone_61 = None
        t_123 = arg216_1.t() ;  arg216_1 = None
        view_382 = _unsafe_view_60.view(2560, 768) ;  _unsafe_view_60 = None
        mm_105 = torch.mm(view_382,t_123) ;  view_382 = t_123 = None
        view_383 = mm_105.view(1, 5, 512, 768) ;  mm_105 = None
        view_384 = getitem_33.view(2560, 384) 
        t_124 = arg211_1.t() ;  arg211_1 = None
        addmm_18 = torch.addmm(arg212_1,view_384,t_124) ;  arg212_1 = view_384 = t_124 = None
        view_385 = addmm_18.view(1, 5, 512, 768) ;  addmm_18 = None
        sigmoid_24 = torch.sigmoid(view_385) ;  view_385 = None
        mul_90 = torch.mul(sigmoid_24,view_383) ;  sigmoid_24 = view_383 = None
        native_layer_norm_default_48 = (torch.nn.functional.layer_norm(add_114,[768],None,None,0.1),) 
        getitem_284 = native_layer_norm_default_48[0]
        t_125 = arg204_1.t() ;  arg204_1 = None
        view_386 = getitem_33.view(2560, 384) 
        mm_106 = torch.mm(view_386,t_125) ;  view_386 = t_125 = None
        view_387 = mm_106.view(1, 5, 512, 1536) ;  mm_106 = None
        split_tensor_55 = torch.split(view_387,768,dim = -1) ;  view_387 = None
        getitem_287 = split_tensor_55[0]
        getitem_288 = split_tensor_55[1];  split_tensor_55 = None
        add_119 = torch.add(getitem_287,1) ;  getitem_287 = None
        mul_91 = torch.mul(getitem_284,add_119) ;  getitem_284 = add_119 = None
        add_120 = torch.add(mul_91,getitem_288) ;  mul_91 = getitem_288 = None
        t_126 = arg205_1.t() ;  arg205_1 = None
        view_388 = add_120.view(2560, 768) ;  add_120 = None
        mm_107 = torch.mm(view_388,t_126) ;  view_388 = t_126 = None
        view_389 = mm_107.view(1, 5, 512, 3072) ;  mm_107 = None
        split_tensor_56 = torch.split(view_389,1536,dim = -1) ;  view_389 = None
        getitem_289 = split_tensor_56[0]
        getitem_290 = split_tensor_56[1];  split_tensor_56 = None
        silu_16 = torch.nn.functional.silu(getitem_289) ;  getitem_289 = None
        mul_92 = torch.mul(silu_16,getitem_290) ;  silu_16 = getitem_290 = None
        view_390 = getitem_33.view(2560, 384) 
        t_127 = arg207_1.t() ;  arg207_1 = None
        addmm_19 = torch.addmm(arg208_1,view_390,t_127) ;  arg208_1 = view_390 = t_127 = None
        view_391 = addmm_19.view(1, 5, 512, 768) ;  addmm_19 = None
        sigmoid_25 = torch.sigmoid(view_391) ;  view_391 = None
        t_128 = arg206_1.t() ;  arg206_1 = None
        view_392 = mul_92.view(2560, 1536) ;  mul_92 = None
        mm_108 = torch.mm(view_392,t_128) ;  view_392 = t_128 = None
        view_393 = mm_108.view(1, 5, 512, 768) ;  mm_108 = None
        mul_93 = torch.mul(sigmoid_25,view_393) ;  sigmoid_25 = view_393 = None
        add_121 = torch.add(mul_90,mul_93) ;  mul_90 = mul_93 = None
        add_122 = torch.add(add_114,add_121) ;  add_114 = add_121 = None
        native_layer_norm_default_49 = (torch.nn.functional.layer_norm(view_140,[256],arg227_1,arg228_1,1e-05),) ;  arg227_1 = arg228_1 = None
        getitem_291 = native_layer_norm_default_49[0]
        t_129 = arg229_1.t() ;  arg229_1 = None
        view_394 = getitem_291.view(262144, 256) ;  getitem_291 = None
        mm_109 = torch.mm(view_394,t_129) ;  view_394 = t_129 = None
        view_395 = mm_109.view(1, 1, 512, 512, 16) ;  mm_109 = None
        native_layer_norm_default_50 = (torch.nn.functional.layer_norm(add_122,[768],None,None,0.1),) 
        getitem_294 = native_layer_norm_default_50[0]
        t_130 = arg223_1.t() ;  arg223_1 = None
        view_396 = getitem_33.view(2560, 384) 
        mm_110 = torch.mm(view_396,t_130) ;  view_396 = t_130 = None
        view_397 = mm_110.view(1, 5, 512, 1536) ;  mm_110 = None
        split_tensor_57 = torch.split(view_397,768,dim = -1) ;  view_397 = None
        getitem_297 = split_tensor_57[0]
        getitem_298 = split_tensor_57[1];  split_tensor_57 = None
        add_123 = torch.add(getitem_297,1) ;  getitem_297 = None
        mul_94 = torch.mul(getitem_294,add_123) ;  getitem_294 = add_123 = None
        add_124 = torch.add(mul_94,getitem_298) ;  mul_94 = getitem_298 = None
        t_131 = arg224_1.t() ;  arg224_1 = None
        view_398 = add_124.view(2560, 768) ;  add_124 = None
        mm_111 = torch.mm(view_398,t_131) ;  view_398 = t_131 = None
        view_399 = mm_111.view(1, 5, 512, 2304) ;  mm_111 = None
        view_400 = view_399.view(1, 5, 512, 16, 144) ;  view_399 = None
        permute_63 = view_400.permute(0, 3, 1, 2, 4) ;  view_400 = None
        split_tensor_58 = torch.split(permute_63,48,dim = -1) ;  permute_63 = None
        getitem_299 = split_tensor_58[0]
        getitem_300 = split_tensor_58[1]
        getitem_301 = split_tensor_58[2];  split_tensor_58 = None
        view_401 = arg217_1.view(1, 16, 1, 1, 48) ;  arg217_1 = None
        add_125 = torch.add(getitem_299,view_401) ;  getitem_299 = view_401 = None
        view_402 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_26 = torch.bitwise_not(view_402) ;  view_402 = None
        masked_fill_26 = view_395.masked_fill(bitwise_not_26,-10000) ;  view_395 = bitwise_not_26 = None
        permute_64 = masked_fill_26.permute(0, 4, 1, 2, 3) ;  masked_fill_26 = None
        view_403 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_27 = torch.bitwise_not(view_403) ;  view_403 = None
        masked_fill_27 = permute_64.masked_fill(bitwise_not_27,-10000) ;  permute_64 = bitwise_not_27 = None
        mul_95 = torch.mul(add_125,0.3799178428257963) ;  add_125 = None
        transpose_10 = torch.transpose(getitem_300,-2,-1) ;  getitem_300 = None
        mul_96 = torch.mul(transpose_10,0.3799178428257963) ;  transpose_10 = None
        expand_54 = mul_95.expand(1, 16, 5, 512, 48) ;  mul_95 = None
        clone_62 = torch.clone(expand_54,memory_format = torch.contiguous_format) ;  expand_54 = None
        _unsafe_view_61 = clone_62.view(80, 512, 48) ;  clone_62 = None
        expand_55 = mul_96.expand(1, 16, 5, 48, 512) ;  mul_96 = None
        clone_63 = torch.clone(expand_55,memory_format = torch.contiguous_format) ;  expand_55 = None
        _unsafe_view_62 = clone_63.view(80, 48, 512) ;  clone_63 = None
        bmm_26 = torch.bmm(_unsafe_view_61,_unsafe_view_62) ;  _unsafe_view_61 = _unsafe_view_62 = None
        view_404 = bmm_26.view(1, 16, 5, 512, 512) ;  bmm_26 = None
        add_126 = torch.add(view_404,masked_fill_27) ;  view_404 = masked_fill_27 = None
        _softmax_10 = torch.softmax(add_126, -1)
        expand_56 = _softmax_10.expand(1, 16, 5, 512, 512) ;  _softmax_10 = None
        view_405 = expand_56.view(80, 512, 512) ;  expand_56 = None
        expand_57 = getitem_301.expand(1, 16, 5, 512, 48) ;  getitem_301 = None
        clone_64 = torch.clone(expand_57,memory_format = torch.contiguous_format) ;  expand_57 = None
        _unsafe_view_63 = clone_64.view(80, 512, 48) ;  clone_64 = None
        bmm_27 = torch.bmm(view_405,_unsafe_view_63) ;  view_405 = _unsafe_view_63 = None
        view_406 = bmm_27.view(1, 16, 5, 512, 48) ;  bmm_27 = None
        permute_65 = view_406.permute(0, 2, 3, 1, 4) ;  view_406 = None
        clone_65 = torch.clone(permute_65,memory_format = torch.contiguous_format) ;  permute_65 = None
        _unsafe_view_64 = clone_65.view(1, 5, 512, 768) ;  clone_65 = None
        t_132 = arg230_1.t() ;  arg230_1 = None
        view_407 = _unsafe_view_64.view(2560, 768) ;  _unsafe_view_64 = None
        mm_112 = torch.mm(view_407,t_132) ;  view_407 = t_132 = None
        view_408 = mm_112.view(1, 5, 512, 768) ;  mm_112 = None
        view_409 = getitem_33.view(2560, 384) 
        t_133 = arg225_1.t() ;  arg225_1 = None
        addmm_20 = torch.addmm(arg226_1,view_409,t_133) ;  arg226_1 = view_409 = t_133 = None
        view_410 = addmm_20.view(1, 5, 512, 768) ;  addmm_20 = None
        sigmoid_26 = torch.sigmoid(view_410) ;  view_410 = None
        mul_97 = torch.mul(sigmoid_26,view_408) ;  sigmoid_26 = view_408 = None
        native_layer_norm_default_51 = (torch.nn.functional.layer_norm(add_122,[768],None,None,0.1),) 
        getitem_302 = native_layer_norm_default_51[0]
        t_134 = arg218_1.t() ;  arg218_1 = None
        view_411 = getitem_33.view(2560, 384) 
        mm_113 = torch.mm(view_411,t_134) ;  view_411 = t_134 = None
        view_412 = mm_113.view(1, 5, 512, 1536) ;  mm_113 = None
        split_tensor_59 = torch.split(view_412,768,dim = -1) ;  view_412 = None
        getitem_305 = split_tensor_59[0]
        getitem_306 = split_tensor_59[1];  split_tensor_59 = None
        add_127 = torch.add(getitem_305,1) ;  getitem_305 = None
        mul_98 = torch.mul(getitem_302,add_127) ;  getitem_302 = add_127 = None
        add_128 = torch.add(mul_98,getitem_306) ;  mul_98 = getitem_306 = None
        t_135 = arg219_1.t() ;  arg219_1 = None
        view_413 = add_128.view(2560, 768) ;  add_128 = None
        mm_114 = torch.mm(view_413,t_135) ;  view_413 = t_135 = None
        view_414 = mm_114.view(1, 5, 512, 3072) ;  mm_114 = None
        split_tensor_60 = torch.split(view_414,1536,dim = -1) ;  view_414 = None
        getitem_307 = split_tensor_60[0]
        getitem_308 = split_tensor_60[1];  split_tensor_60 = None
        silu_17 = torch.nn.functional.silu(getitem_307) ;  getitem_307 = None
        mul_99 = torch.mul(silu_17,getitem_308) ;  silu_17 = getitem_308 = None
        view_415 = getitem_33.view(2560, 384) 
        t_136 = arg221_1.t() ;  arg221_1 = None
        addmm_21 = torch.addmm(arg222_1,view_415,t_136) ;  arg222_1 = view_415 = t_136 = None
        view_416 = addmm_21.view(1, 5, 512, 768) ;  addmm_21 = None
        sigmoid_27 = torch.sigmoid(view_416) ;  view_416 = None
        t_137 = arg220_1.t() ;  arg220_1 = None
        view_417 = mul_99.view(2560, 1536) ;  mul_99 = None
        mm_115 = torch.mm(view_417,t_137) ;  view_417 = t_137 = None
        view_418 = mm_115.view(1, 5, 512, 768) ;  mm_115 = None
        mul_100 = torch.mul(sigmoid_27,view_418) ;  sigmoid_27 = view_418 = None
        add_129 = torch.add(mul_97,mul_100) ;  mul_97 = mul_100 = None
        add_130 = torch.add(add_122,add_129) ;  add_122 = add_129 = None
        native_layer_norm_default_52 = (torch.nn.functional.layer_norm(view_140,[256],arg241_1,arg242_1,1e-05),) ;  arg241_1 = arg242_1 = None
        getitem_309 = native_layer_norm_default_52[0]
        t_138 = arg243_1.t() ;  arg243_1 = None
        view_419 = getitem_309.view(262144, 256) ;  getitem_309 = None
        mm_116 = torch.mm(view_419,t_138) ;  view_419 = t_138 = None
        view_420 = mm_116.view(1, 1, 512, 512, 16) ;  mm_116 = None
        native_layer_norm_default_53 = (torch.nn.functional.layer_norm(add_130,[768],None,None,0.1),) 
        getitem_312 = native_layer_norm_default_53[0]
        t_139 = arg237_1.t() ;  arg237_1 = None
        view_421 = getitem_33.view(2560, 384) 
        mm_117 = torch.mm(view_421,t_139) ;  view_421 = t_139 = None
        view_422 = mm_117.view(1, 5, 512, 1536) ;  mm_117 = None
        split_tensor_61 = torch.split(view_422,768,dim = -1) ;  view_422 = None
        getitem_315 = split_tensor_61[0]
        getitem_316 = split_tensor_61[1];  split_tensor_61 = None
        add_131 = torch.add(getitem_315,1) ;  getitem_315 = None
        mul_101 = torch.mul(getitem_312,add_131) ;  getitem_312 = add_131 = None
        add_132 = torch.add(mul_101,getitem_316) ;  mul_101 = getitem_316 = None
        t_140 = arg238_1.t() ;  arg238_1 = None
        view_423 = add_132.view(2560, 768) ;  add_132 = None
        mm_118 = torch.mm(view_423,t_140) ;  view_423 = t_140 = None
        view_424 = mm_118.view(1, 5, 512, 2304) ;  mm_118 = None
        view_425 = view_424.view(1, 5, 512, 16, 144) ;  view_424 = None
        permute_66 = view_425.permute(0, 3, 1, 2, 4) ;  view_425 = None
        split_tensor_62 = torch.split(permute_66,48,dim = -1) ;  permute_66 = None
        getitem_317 = split_tensor_62[0]
        getitem_318 = split_tensor_62[1]
        getitem_319 = split_tensor_62[2];  split_tensor_62 = None
        view_426 = arg231_1.view(1, 16, 1, 1, 48) ;  arg231_1 = None
        add_133 = torch.add(getitem_317,view_426) ;  getitem_317 = view_426 = None
        view_427 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_28 = torch.bitwise_not(view_427) ;  view_427 = None
        masked_fill_28 = view_420.masked_fill(bitwise_not_28,-10000) ;  view_420 = bitwise_not_28 = None
        permute_67 = masked_fill_28.permute(0, 4, 1, 2, 3) ;  masked_fill_28 = None
        view_428 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_29 = torch.bitwise_not(view_428) ;  view_428 = None
        masked_fill_29 = permute_67.masked_fill(bitwise_not_29,-10000) ;  permute_67 = bitwise_not_29 = None
        mul_102 = torch.mul(add_133,0.3799178428257963) ;  add_133 = None
        transpose_11 = torch.transpose(getitem_318,-2,-1) ;  getitem_318 = None
        mul_103 = torch.mul(transpose_11,0.3799178428257963) ;  transpose_11 = None
        expand_58 = mul_102.expand(1, 16, 5, 512, 48) ;  mul_102 = None
        clone_66 = torch.clone(expand_58,memory_format = torch.contiguous_format) ;  expand_58 = None
        _unsafe_view_65 = clone_66.view(80, 512, 48) ;  clone_66 = None
        expand_59 = mul_103.expand(1, 16, 5, 48, 512) ;  mul_103 = None
        clone_67 = torch.clone(expand_59,memory_format = torch.contiguous_format) ;  expand_59 = None
        _unsafe_view_66 = clone_67.view(80, 48, 512) ;  clone_67 = None
        bmm_28 = torch.bmm(_unsafe_view_65,_unsafe_view_66) ;  _unsafe_view_65 = _unsafe_view_66 = None
        view_429 = bmm_28.view(1, 16, 5, 512, 512) ;  bmm_28 = None
        add_134 = torch.add(view_429,masked_fill_29) ;  view_429 = masked_fill_29 = None
        _softmax_11 = torch.softmax(add_134, -1)
        expand_60 = _softmax_11.expand(1, 16, 5, 512, 512) ;  _softmax_11 = None
        view_430 = expand_60.view(80, 512, 512) ;  expand_60 = None
        expand_61 = getitem_319.expand(1, 16, 5, 512, 48) ;  getitem_319 = None
        clone_68 = torch.clone(expand_61,memory_format = torch.contiguous_format) ;  expand_61 = None
        _unsafe_view_67 = clone_68.view(80, 512, 48) ;  clone_68 = None
        bmm_29 = torch.bmm(view_430,_unsafe_view_67) ;  view_430 = _unsafe_view_67 = None
        view_431 = bmm_29.view(1, 16, 5, 512, 48) ;  bmm_29 = None
        permute_68 = view_431.permute(0, 2, 3, 1, 4) ;  view_431 = None
        clone_69 = torch.clone(permute_68,memory_format = torch.contiguous_format) ;  permute_68 = None
        _unsafe_view_68 = clone_69.view(1, 5, 512, 768) ;  clone_69 = None
        t_141 = arg244_1.t() ;  arg244_1 = None
        view_432 = _unsafe_view_68.view(2560, 768) ;  _unsafe_view_68 = None
        mm_119 = torch.mm(view_432,t_141) ;  view_432 = t_141 = None
        view_433 = mm_119.view(1, 5, 512, 768) ;  mm_119 = None
        view_434 = getitem_33.view(2560, 384) 
        t_142 = arg239_1.t() ;  arg239_1 = None
        addmm_22 = torch.addmm(arg240_1,view_434,t_142) ;  arg240_1 = view_434 = t_142 = None
        view_435 = addmm_22.view(1, 5, 512, 768) ;  addmm_22 = None
        sigmoid_28 = torch.sigmoid(view_435) ;  view_435 = None
        mul_104 = torch.mul(sigmoid_28,view_433) ;  sigmoid_28 = view_433 = None
        native_layer_norm_default_54 = (torch.nn.functional.layer_norm(add_130,[768],None,None,0.1),) 
        getitem_320 = native_layer_norm_default_54[0]
        t_143 = arg232_1.t() ;  arg232_1 = None
        view_436 = getitem_33.view(2560, 384) 
        mm_120 = torch.mm(view_436,t_143) ;  view_436 = t_143 = None
        view_437 = mm_120.view(1, 5, 512, 1536) ;  mm_120 = None
        split_tensor_63 = torch.split(view_437,768,dim = -1) ;  view_437 = None
        getitem_323 = split_tensor_63[0]
        getitem_324 = split_tensor_63[1];  split_tensor_63 = None
        add_135 = torch.add(getitem_323,1) ;  getitem_323 = None
        mul_105 = torch.mul(getitem_320,add_135) ;  getitem_320 = add_135 = None
        add_136 = torch.add(mul_105,getitem_324) ;  mul_105 = getitem_324 = None
        t_144 = arg233_1.t() ;  arg233_1 = None
        view_438 = add_136.view(2560, 768) ;  add_136 = None
        mm_121 = torch.mm(view_438,t_144) ;  view_438 = t_144 = None
        view_439 = mm_121.view(1, 5, 512, 3072) ;  mm_121 = None
        split_tensor_64 = torch.split(view_439,1536,dim = -1) ;  view_439 = None
        getitem_325 = split_tensor_64[0]
        getitem_326 = split_tensor_64[1];  split_tensor_64 = None
        silu_18 = torch.nn.functional.silu(getitem_325) ;  getitem_325 = None
        mul_106 = torch.mul(silu_18,getitem_326) ;  silu_18 = getitem_326 = None
        view_440 = getitem_33.view(2560, 384) 
        t_145 = arg235_1.t() ;  arg235_1 = None
        addmm_23 = torch.addmm(arg236_1,view_440,t_145) ;  arg236_1 = view_440 = t_145 = None
        view_441 = addmm_23.view(1, 5, 512, 768) ;  addmm_23 = None
        sigmoid_29 = torch.sigmoid(view_441) ;  view_441 = None
        t_146 = arg234_1.t() ;  arg234_1 = None
        view_442 = mul_106.view(2560, 1536) ;  mul_106 = None
        mm_122 = torch.mm(view_442,t_146) ;  view_442 = t_146 = None
        view_443 = mm_122.view(1, 5, 512, 768) ;  mm_122 = None
        mul_107 = torch.mul(sigmoid_29,view_443) ;  sigmoid_29 = view_443 = None
        add_137 = torch.add(mul_104,mul_107) ;  mul_104 = mul_107 = None
        add_138 = torch.add(add_130,add_137) ;  add_130 = add_137 = None
        native_layer_norm_default_55 = (torch.nn.functional.layer_norm(view_140,[256],arg255_1,arg256_1,1e-05),) ;  arg255_1 = arg256_1 = None
        getitem_327 = native_layer_norm_default_55[0]
        t_147 = arg257_1.t() ;  arg257_1 = None
        view_444 = getitem_327.view(262144, 256) ;  getitem_327 = None
        mm_123 = torch.mm(view_444,t_147) ;  view_444 = t_147 = None
        view_445 = mm_123.view(1, 1, 512, 512, 16) ;  mm_123 = None
        native_layer_norm_default_56 = (torch.nn.functional.layer_norm(add_138,[768],None,None,0.1),) 
        getitem_330 = native_layer_norm_default_56[0]
        t_148 = arg251_1.t() ;  arg251_1 = None
        view_446 = getitem_33.view(2560, 384) 
        mm_124 = torch.mm(view_446,t_148) ;  view_446 = t_148 = None
        view_447 = mm_124.view(1, 5, 512, 1536) ;  mm_124 = None
        split_tensor_65 = torch.split(view_447,768,dim = -1) ;  view_447 = None
        getitem_333 = split_tensor_65[0]
        getitem_334 = split_tensor_65[1];  split_tensor_65 = None
        add_139 = torch.add(getitem_333,1) ;  getitem_333 = None
        mul_108 = torch.mul(getitem_330,add_139) ;  getitem_330 = add_139 = None
        add_140 = torch.add(mul_108,getitem_334) ;  mul_108 = getitem_334 = None
        t_149 = arg252_1.t() ;  arg252_1 = None
        view_448 = add_140.view(2560, 768) ;  add_140 = None
        mm_125 = torch.mm(view_448,t_149) ;  view_448 = t_149 = None
        view_449 = mm_125.view(1, 5, 512, 2304) ;  mm_125 = None
        view_450 = view_449.view(1, 5, 512, 16, 144) ;  view_449 = None
        permute_69 = view_450.permute(0, 3, 1, 2, 4) ;  view_450 = None
        split_tensor_66 = torch.split(permute_69,48,dim = -1) ;  permute_69 = None
        getitem_335 = split_tensor_66[0]
        getitem_336 = split_tensor_66[1]
        getitem_337 = split_tensor_66[2];  split_tensor_66 = None
        view_451 = arg245_1.view(1, 16, 1, 1, 48) ;  arg245_1 = None
        add_141 = torch.add(getitem_335,view_451) ;  getitem_335 = view_451 = None
        view_452 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_30 = torch.bitwise_not(view_452) ;  view_452 = None
        masked_fill_30 = view_445.masked_fill(bitwise_not_30,-10000) ;  view_445 = bitwise_not_30 = None
        permute_70 = masked_fill_30.permute(0, 4, 1, 2, 3) ;  masked_fill_30 = None
        view_453 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_31 = torch.bitwise_not(view_453) ;  view_453 = None
        masked_fill_31 = permute_70.masked_fill(bitwise_not_31,-10000) ;  permute_70 = bitwise_not_31 = None
        mul_109 = torch.mul(add_141,0.3799178428257963) ;  add_141 = None
        transpose_12 = torch.transpose(getitem_336,-2,-1) ;  getitem_336 = None
        mul_110 = torch.mul(transpose_12,0.3799178428257963) ;  transpose_12 = None
        expand_62 = mul_109.expand(1, 16, 5, 512, 48) ;  mul_109 = None
        clone_70 = torch.clone(expand_62,memory_format = torch.contiguous_format) ;  expand_62 = None
        _unsafe_view_69 = clone_70.view(80, 512, 48) ;  clone_70 = None
        expand_63 = mul_110.expand(1, 16, 5, 48, 512) ;  mul_110 = None
        clone_71 = torch.clone(expand_63,memory_format = torch.contiguous_format) ;  expand_63 = None
        _unsafe_view_70 = clone_71.view(80, 48, 512) ;  clone_71 = None
        bmm_30 = torch.bmm(_unsafe_view_69,_unsafe_view_70) ;  _unsafe_view_69 = _unsafe_view_70 = None
        view_454 = bmm_30.view(1, 16, 5, 512, 512) ;  bmm_30 = None
        add_142 = torch.add(view_454,masked_fill_31) ;  view_454 = masked_fill_31 = None
        _softmax_12 = torch.softmax(add_142, -1)
        expand_64 = _softmax_12.expand(1, 16, 5, 512, 512) ;  _softmax_12 = None
        view_455 = expand_64.view(80, 512, 512) ;  expand_64 = None
        expand_65 = getitem_337.expand(1, 16, 5, 512, 48) ;  getitem_337 = None
        clone_72 = torch.clone(expand_65,memory_format = torch.contiguous_format) ;  expand_65 = None
        _unsafe_view_71 = clone_72.view(80, 512, 48) ;  clone_72 = None
        bmm_31 = torch.bmm(view_455,_unsafe_view_71) ;  view_455 = _unsafe_view_71 = None
        view_456 = bmm_31.view(1, 16, 5, 512, 48) ;  bmm_31 = None
        permute_71 = view_456.permute(0, 2, 3, 1, 4) ;  view_456 = None
        clone_73 = torch.clone(permute_71,memory_format = torch.contiguous_format) ;  permute_71 = None
        _unsafe_view_72 = clone_73.view(1, 5, 512, 768) ;  clone_73 = None
        t_150 = arg258_1.t() ;  arg258_1 = None
        view_457 = _unsafe_view_72.view(2560, 768) ;  _unsafe_view_72 = None
        mm_126 = torch.mm(view_457,t_150) ;  view_457 = t_150 = None
        view_458 = mm_126.view(1, 5, 512, 768) ;  mm_126 = None
        view_459 = getitem_33.view(2560, 384) 
        t_151 = arg253_1.t() ;  arg253_1 = None
        addmm_24 = torch.addmm(arg254_1,view_459,t_151) ;  arg254_1 = view_459 = t_151 = None
        view_460 = addmm_24.view(1, 5, 512, 768) ;  addmm_24 = None
        sigmoid_30 = torch.sigmoid(view_460) ;  view_460 = None
        mul_111 = torch.mul(sigmoid_30,view_458) ;  sigmoid_30 = view_458 = None
        native_layer_norm_default_57 = (torch.nn.functional.layer_norm(add_138,[768],None,None,0.1),) 
        getitem_338 = native_layer_norm_default_57[0]
        t_152 = arg246_1.t() ;  arg246_1 = None
        view_461 = getitem_33.view(2560, 384) 
        mm_127 = torch.mm(view_461,t_152) ;  view_461 = t_152 = None
        view_462 = mm_127.view(1, 5, 512, 1536) ;  mm_127 = None
        split_tensor_67 = torch.split(view_462,768,dim = -1) ;  view_462 = None
        getitem_341 = split_tensor_67[0]
        getitem_342 = split_tensor_67[1];  split_tensor_67 = None
        add_143 = torch.add(getitem_341,1) ;  getitem_341 = None
        mul_112 = torch.mul(getitem_338,add_143) ;  getitem_338 = add_143 = None
        add_144 = torch.add(mul_112,getitem_342) ;  mul_112 = getitem_342 = None
        t_153 = arg247_1.t() ;  arg247_1 = None
        view_463 = add_144.view(2560, 768) ;  add_144 = None
        mm_128 = torch.mm(view_463,t_153) ;  view_463 = t_153 = None
        view_464 = mm_128.view(1, 5, 512, 3072) ;  mm_128 = None
        split_tensor_68 = torch.split(view_464,1536,dim = -1) ;  view_464 = None
        getitem_343 = split_tensor_68[0]
        getitem_344 = split_tensor_68[1];  split_tensor_68 = None
        silu_19 = torch.nn.functional.silu(getitem_343) ;  getitem_343 = None
        mul_113 = torch.mul(silu_19,getitem_344) ;  silu_19 = getitem_344 = None
        view_465 = getitem_33.view(2560, 384) 
        t_154 = arg249_1.t() ;  arg249_1 = None
        addmm_25 = torch.addmm(arg250_1,view_465,t_154) ;  arg250_1 = view_465 = t_154 = None
        view_466 = addmm_25.view(1, 5, 512, 768) ;  addmm_25 = None
        sigmoid_31 = torch.sigmoid(view_466) ;  view_466 = None
        t_155 = arg248_1.t() ;  arg248_1 = None
        view_467 = mul_113.view(2560, 1536) ;  mul_113 = None
        mm_129 = torch.mm(view_467,t_155) ;  view_467 = t_155 = None
        view_468 = mm_129.view(1, 5, 512, 768) ;  mm_129 = None
        mul_114 = torch.mul(sigmoid_31,view_468) ;  sigmoid_31 = view_468 = None
        add_145 = torch.add(mul_111,mul_114) ;  mul_111 = mul_114 = None
        add_146 = torch.add(add_138,add_145) ;  add_138 = add_145 = None
        native_layer_norm_default_58 = (torch.nn.functional.layer_norm(view_140,[256],arg269_1,arg270_1,1e-05),) ;  arg269_1 = arg270_1 = None
        getitem_345 = native_layer_norm_default_58[0]
        t_156 = arg271_1.t() ;  arg271_1 = None
        view_469 = getitem_345.view(262144, 256) ;  getitem_345 = None
        mm_130 = torch.mm(view_469,t_156) ;  view_469 = t_156 = None
        view_470 = mm_130.view(1, 1, 512, 512, 16) ;  mm_130 = None
        native_layer_norm_default_59 = (torch.nn.functional.layer_norm(add_146,[768],None,None,0.1),) 
        getitem_348 = native_layer_norm_default_59[0]
        t_157 = arg265_1.t() ;  arg265_1 = None
        view_471 = getitem_33.view(2560, 384) 
        mm_131 = torch.mm(view_471,t_157) ;  view_471 = t_157 = None
        view_472 = mm_131.view(1, 5, 512, 1536) ;  mm_131 = None
        split_tensor_69 = torch.split(view_472,768,dim = -1) ;  view_472 = None
        getitem_351 = split_tensor_69[0]
        getitem_352 = split_tensor_69[1];  split_tensor_69 = None
        add_147 = torch.add(getitem_351,1) ;  getitem_351 = None
        mul_115 = torch.mul(getitem_348,add_147) ;  getitem_348 = add_147 = None
        add_148 = torch.add(mul_115,getitem_352) ;  mul_115 = getitem_352 = None
        t_158 = arg266_1.t() ;  arg266_1 = None
        view_473 = add_148.view(2560, 768) ;  add_148 = None
        mm_132 = torch.mm(view_473,t_158) ;  view_473 = t_158 = None
        view_474 = mm_132.view(1, 5, 512, 2304) ;  mm_132 = None
        view_475 = view_474.view(1, 5, 512, 16, 144) ;  view_474 = None
        permute_72 = view_475.permute(0, 3, 1, 2, 4) ;  view_475 = None
        split_tensor_70 = torch.split(permute_72,48,dim = -1) ;  permute_72 = None
        getitem_353 = split_tensor_70[0]
        getitem_354 = split_tensor_70[1]
        getitem_355 = split_tensor_70[2];  split_tensor_70 = None
        view_476 = arg259_1.view(1, 16, 1, 1, 48) ;  arg259_1 = None
        add_149 = torch.add(getitem_353,view_476) ;  getitem_353 = view_476 = None
        view_477 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_32 = torch.bitwise_not(view_477) ;  view_477 = None
        masked_fill_32 = view_470.masked_fill(bitwise_not_32,-10000) ;  view_470 = bitwise_not_32 = None
        permute_73 = masked_fill_32.permute(0, 4, 1, 2, 3) ;  masked_fill_32 = None
        view_478 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_33 = torch.bitwise_not(view_478) ;  view_478 = None
        masked_fill_33 = permute_73.masked_fill(bitwise_not_33,-10000) ;  permute_73 = bitwise_not_33 = None
        mul_116 = torch.mul(add_149,0.3799178428257963) ;  add_149 = None
        transpose_13 = torch.transpose(getitem_354,-2,-1) ;  getitem_354 = None
        mul_117 = torch.mul(transpose_13,0.3799178428257963) ;  transpose_13 = None
        expand_66 = mul_116.expand(1, 16, 5, 512, 48) ;  mul_116 = None
        clone_74 = torch.clone(expand_66,memory_format = torch.contiguous_format) ;  expand_66 = None
        _unsafe_view_73 = clone_74.view(80, 512, 48) ;  clone_74 = None
        expand_67 = mul_117.expand(1, 16, 5, 48, 512) ;  mul_117 = None
        clone_75 = torch.clone(expand_67,memory_format = torch.contiguous_format) ;  expand_67 = None
        _unsafe_view_74 = clone_75.view(80, 48, 512) ;  clone_75 = None
        bmm_32 = torch.bmm(_unsafe_view_73,_unsafe_view_74) ;  _unsafe_view_73 = _unsafe_view_74 = None
        view_479 = bmm_32.view(1, 16, 5, 512, 512) ;  bmm_32 = None
        add_150 = torch.add(view_479,masked_fill_33) ;  view_479 = masked_fill_33 = None
        _softmax_13 = torch.softmax(add_150, -1)
        expand_68 = _softmax_13.expand(1, 16, 5, 512, 512) ;  _softmax_13 = None
        view_480 = expand_68.view(80, 512, 512) ;  expand_68 = None
        expand_69 = getitem_355.expand(1, 16, 5, 512, 48) ;  getitem_355 = None
        clone_76 = torch.clone(expand_69,memory_format = torch.contiguous_format) ;  expand_69 = None
        _unsafe_view_75 = clone_76.view(80, 512, 48) ;  clone_76 = None
        bmm_33 = torch.bmm(view_480,_unsafe_view_75) ;  view_480 = _unsafe_view_75 = None
        view_481 = bmm_33.view(1, 16, 5, 512, 48) ;  bmm_33 = None
        permute_74 = view_481.permute(0, 2, 3, 1, 4) ;  view_481 = None
        clone_77 = torch.clone(permute_74,memory_format = torch.contiguous_format) ;  permute_74 = None
        _unsafe_view_76 = clone_77.view(1, 5, 512, 768) ;  clone_77 = None
        t_159 = arg272_1.t() ;  arg272_1 = None
        view_482 = _unsafe_view_76.view(2560, 768) ;  _unsafe_view_76 = None
        mm_133 = torch.mm(view_482,t_159) ;  view_482 = t_159 = None
        view_483 = mm_133.view(1, 5, 512, 768) ;  mm_133 = None
        view_484 = getitem_33.view(2560, 384) 
        t_160 = arg267_1.t() ;  arg267_1 = None
        addmm_26 = torch.addmm(arg268_1,view_484,t_160) ;  arg268_1 = view_484 = t_160 = None
        view_485 = addmm_26.view(1, 5, 512, 768) ;  addmm_26 = None
        sigmoid_32 = torch.sigmoid(view_485) ;  view_485 = None
        mul_118 = torch.mul(sigmoid_32,view_483) ;  sigmoid_32 = view_483 = None
        native_layer_norm_default_60 = (torch.nn.functional.layer_norm(add_146,[768],None,None,0.1),) 
        getitem_356 = native_layer_norm_default_60[0]
        t_161 = arg260_1.t() ;  arg260_1 = None
        view_486 = getitem_33.view(2560, 384) 
        mm_134 = torch.mm(view_486,t_161) ;  view_486 = t_161 = None
        view_487 = mm_134.view(1, 5, 512, 1536) ;  mm_134 = None
        split_tensor_71 = torch.split(view_487,768,dim = -1) ;  view_487 = None
        getitem_359 = split_tensor_71[0]
        getitem_360 = split_tensor_71[1];  split_tensor_71 = None
        add_151 = torch.add(getitem_359,1) ;  getitem_359 = None
        mul_119 = torch.mul(getitem_356,add_151) ;  getitem_356 = add_151 = None
        add_152 = torch.add(mul_119,getitem_360) ;  mul_119 = getitem_360 = None
        t_162 = arg261_1.t() ;  arg261_1 = None
        view_488 = add_152.view(2560, 768) ;  add_152 = None
        mm_135 = torch.mm(view_488,t_162) ;  view_488 = t_162 = None
        view_489 = mm_135.view(1, 5, 512, 3072) ;  mm_135 = None
        split_tensor_72 = torch.split(view_489,1536,dim = -1) ;  view_489 = None
        getitem_361 = split_tensor_72[0]
        getitem_362 = split_tensor_72[1];  split_tensor_72 = None
        silu_20 = torch.nn.functional.silu(getitem_361) ;  getitem_361 = None
        mul_120 = torch.mul(silu_20,getitem_362) ;  silu_20 = getitem_362 = None
        view_490 = getitem_33.view(2560, 384) 
        t_163 = arg263_1.t() ;  arg263_1 = None
        addmm_27 = torch.addmm(arg264_1,view_490,t_163) ;  arg264_1 = view_490 = t_163 = None
        view_491 = addmm_27.view(1, 5, 512, 768) ;  addmm_27 = None
        sigmoid_33 = torch.sigmoid(view_491) ;  view_491 = None
        t_164 = arg262_1.t() ;  arg262_1 = None
        view_492 = mul_120.view(2560, 1536) ;  mul_120 = None
        mm_136 = torch.mm(view_492,t_164) ;  view_492 = t_164 = None
        view_493 = mm_136.view(1, 5, 512, 768) ;  mm_136 = None
        mul_121 = torch.mul(sigmoid_33,view_493) ;  sigmoid_33 = view_493 = None
        add_153 = torch.add(mul_118,mul_121) ;  mul_118 = mul_121 = None
        add_154 = torch.add(add_146,add_153) ;  add_146 = add_153 = None
        native_layer_norm_default_61 = (torch.nn.functional.layer_norm(view_140,[256],arg283_1,arg284_1,1e-05),) ;  arg283_1 = arg284_1 = None
        getitem_363 = native_layer_norm_default_61[0]
        t_165 = arg285_1.t() ;  arg285_1 = None
        view_494 = getitem_363.view(262144, 256) ;  getitem_363 = None
        mm_137 = torch.mm(view_494,t_165) ;  view_494 = t_165 = None
        view_495 = mm_137.view(1, 1, 512, 512, 16) ;  mm_137 = None
        native_layer_norm_default_62 = (torch.nn.functional.layer_norm(add_154,[768],None,None,0.1),) 
        getitem_366 = native_layer_norm_default_62[0]
        t_166 = arg279_1.t() ;  arg279_1 = None
        view_496 = getitem_33.view(2560, 384) 
        mm_138 = torch.mm(view_496,t_166) ;  view_496 = t_166 = None
        view_497 = mm_138.view(1, 5, 512, 1536) ;  mm_138 = None
        split_tensor_73 = torch.split(view_497,768,dim = -1) ;  view_497 = None
        getitem_369 = split_tensor_73[0]
        getitem_370 = split_tensor_73[1];  split_tensor_73 = None
        add_155 = torch.add(getitem_369,1) ;  getitem_369 = None
        mul_122 = torch.mul(getitem_366,add_155) ;  getitem_366 = add_155 = None
        add_156 = torch.add(mul_122,getitem_370) ;  mul_122 = getitem_370 = None
        t_167 = arg280_1.t() ;  arg280_1 = None
        view_498 = add_156.view(2560, 768) ;  add_156 = None
        mm_139 = torch.mm(view_498,t_167) ;  view_498 = t_167 = None
        view_499 = mm_139.view(1, 5, 512, 2304) ;  mm_139 = None
        view_500 = view_499.view(1, 5, 512, 16, 144) ;  view_499 = None
        permute_75 = view_500.permute(0, 3, 1, 2, 4) ;  view_500 = None
        split_tensor_74 = torch.split(permute_75,48,dim = -1) ;  permute_75 = None
        getitem_371 = split_tensor_74[0]
        getitem_372 = split_tensor_74[1]
        getitem_373 = split_tensor_74[2];  split_tensor_74 = None
        view_501 = arg273_1.view(1, 16, 1, 1, 48) ;  arg273_1 = None
        add_157 = torch.add(getitem_371,view_501) ;  getitem_371 = view_501 = None
        view_502 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_34 = torch.bitwise_not(view_502) ;  view_502 = None
        masked_fill_34 = view_495.masked_fill(bitwise_not_34,-10000) ;  view_495 = bitwise_not_34 = None
        permute_76 = masked_fill_34.permute(0, 4, 1, 2, 3) ;  masked_fill_34 = None
        view_503 = bitwise_and_1.view(1, 1, 1, 512, 512) 
        bitwise_not_35 = torch.bitwise_not(view_503) ;  view_503 = None
        masked_fill_35 = permute_76.masked_fill(bitwise_not_35,-10000) ;  permute_76 = bitwise_not_35 = None
        mul_123 = torch.mul(add_157,0.3799178428257963) ;  add_157 = None
        transpose_14 = torch.transpose(getitem_372,-2,-1) ;  getitem_372 = None
        mul_124 = torch.mul(transpose_14,0.3799178428257963) ;  transpose_14 = None
        expand_70 = mul_123.expand(1, 16, 5, 512, 48) ;  mul_123 = None
        clone_78 = torch.clone(expand_70,memory_format = torch.contiguous_format) ;  expand_70 = None
        _unsafe_view_77 = clone_78.view(80, 512, 48) ;  clone_78 = None
        expand_71 = mul_124.expand(1, 16, 5, 48, 512) ;  mul_124 = None
        clone_79 = torch.clone(expand_71,memory_format = torch.contiguous_format) ;  expand_71 = None
        _unsafe_view_78 = clone_79.view(80, 48, 512) ;  clone_79 = None
        bmm_34 = torch.bmm(_unsafe_view_77,_unsafe_view_78) ;  _unsafe_view_77 = _unsafe_view_78 = None
        view_504 = bmm_34.view(1, 16, 5, 512, 512) ;  bmm_34 = None
        add_158 = torch.add(view_504,masked_fill_35) ;  view_504 = masked_fill_35 = None
        _softmax_14 = torch.softmax(add_158, -1)
        expand_72 = _softmax_14.expand(1, 16, 5, 512, 512) ;  _softmax_14 = None
        view_505 = expand_72.view(80, 512, 512) ;  expand_72 = None
        expand_73 = getitem_373.expand(1, 16, 5, 512, 48) ;  getitem_373 = None
        clone_80 = torch.clone(expand_73,memory_format = torch.contiguous_format) ;  expand_73 = None
        _unsafe_view_79 = clone_80.view(80, 512, 48) ;  clone_80 = None
        bmm_35 = torch.bmm(view_505,_unsafe_view_79) ;  view_505 = _unsafe_view_79 = None
        view_506 = bmm_35.view(1, 16, 5, 512, 48) ;  bmm_35 = None
        permute_77 = view_506.permute(0, 2, 3, 1, 4) ;  view_506 = None
        clone_81 = torch.clone(permute_77,memory_format = torch.contiguous_format) ;  permute_77 = None
        _unsafe_view_80 = clone_81.view(1, 5, 512, 768) ;  clone_81 = None
        t_168 = arg286_1.t() ;  arg286_1 = None
        view_507 = _unsafe_view_80.view(2560, 768) ;  _unsafe_view_80 = None
        mm_140 = torch.mm(view_507,t_168) ;  view_507 = t_168 = None
        view_508 = mm_140.view(1, 5, 512, 768) ;  mm_140 = None
        view_509 = getitem_33.view(2560, 384) 
        t_169 = arg281_1.t() ;  arg281_1 = None
        addmm_28 = torch.addmm(arg282_1,view_509,t_169) ;  arg282_1 = view_509 = t_169 = None
        view_510 = addmm_28.view(1, 5, 512, 768) ;  addmm_28 = None
        sigmoid_34 = torch.sigmoid(view_510) ;  view_510 = None
        mul_125 = torch.mul(sigmoid_34,view_508) ;  sigmoid_34 = view_508 = None
        native_layer_norm_default_63 = (torch.nn.functional.layer_norm(add_154,[768],None,None,0.1),) 
        getitem_374 = native_layer_norm_default_63[0]
        t_170 = arg274_1.t() ;  arg274_1 = None
        view_511 = getitem_33.view(2560, 384) 
        mm_141 = torch.mm(view_511,t_170) ;  view_511 = t_170 = None
        view_512 = mm_141.view(1, 5, 512, 1536) ;  mm_141 = None
        split_tensor_75 = torch.split(view_512,768,dim = -1) ;  view_512 = None
        getitem_377 = split_tensor_75[0]
        getitem_378 = split_tensor_75[1];  split_tensor_75 = None
        add_159 = torch.add(getitem_377,1) ;  getitem_377 = None
        mul_126 = torch.mul(getitem_374,add_159) ;  getitem_374 = add_159 = None
        add_160 = torch.add(mul_126,getitem_378) ;  mul_126 = getitem_378 = None
        t_171 = arg275_1.t() ;  arg275_1 = None
        view_513 = add_160.view(2560, 768) ;  add_160 = None
        mm_142 = torch.mm(view_513,t_171) ;  view_513 = t_171 = None
        view_514 = mm_142.view(1, 5, 512, 3072) ;  mm_142 = None
        split_tensor_76 = torch.split(view_514,1536,dim = -1) ;  view_514 = None
        getitem_379 = split_tensor_76[0]
        getitem_380 = split_tensor_76[1];  split_tensor_76 = None
        silu_21 = torch.nn.functional.silu(getitem_379) ;  getitem_379 = None
        mul_127 = torch.mul(silu_21,getitem_380) ;  silu_21 = getitem_380 = None
        view_515 = getitem_33.view(2560, 384) 
        t_172 = arg277_1.t() ;  arg277_1 = None
        addmm_29 = torch.addmm(arg278_1,view_515,t_172) ;  arg278_1 = view_515 = t_172 = None
        view_516 = addmm_29.view(1, 5, 512, 768) ;  addmm_29 = None
        sigmoid_35 = torch.sigmoid(view_516) ;  view_516 = None
        t_173 = arg276_1.t() ;  arg276_1 = None
        view_517 = mul_127.view(2560, 1536) ;  mul_127 = None
        mm_143 = torch.mm(view_517,t_173) ;  view_517 = t_173 = None
        view_518 = mm_143.view(1, 5, 512, 768) ;  mm_143 = None
        mul_128 = torch.mul(sigmoid_35,view_518) ;  sigmoid_35 = view_518 = None
        add_161 = torch.add(mul_125,mul_128) ;  mul_125 = mul_128 = None
        add_162 = torch.add(add_154,add_161) ;  add_154 = add_161 = None
        native_layer_norm_default_64 = (torch.nn.functional.layer_norm(view_140,[256],arg297_1,arg298_1,1e-05),) ;  view_140 = arg297_1 = arg298_1 = None
        getitem_381 = native_layer_norm_default_64[0]
        t_174 = arg299_1.t() ;  arg299_1 = None
        view_519 = getitem_381.view(262144, 256) ;  getitem_381 = None
        mm_144 = torch.mm(view_519,t_174) ;  view_519 = t_174 = None
        view_520 = mm_144.view(1, 1, 512, 512, 16) ;  mm_144 = None
        native_layer_norm_default_65 = (torch.nn.functional.layer_norm(add_162,[768],None,None,0.1),) 
        getitem_384 = native_layer_norm_default_65[0]
        t_175 = arg293_1.t() ;  arg293_1 = None
        view_521 = getitem_33.view(2560, 384) 
        mm_145 = torch.mm(view_521,t_175) ;  view_521 = t_175 = None
        view_522 = mm_145.view(1, 5, 512, 1536) ;  mm_145 = None
        split_tensor_77 = torch.split(view_522,768,dim = -1) ;  view_522 = None
        getitem_387 = split_tensor_77[0]
        getitem_388 = split_tensor_77[1];  split_tensor_77 = None
        add_163 = torch.add(getitem_387,1) ;  getitem_387 = None
        mul_129 = torch.mul(getitem_384,add_163) ;  getitem_384 = add_163 = None
        add_164 = torch.add(mul_129,getitem_388) ;  mul_129 = getitem_388 = None
        t_176 = arg294_1.t() ;  arg294_1 = None
        view_523 = add_164.view(2560, 768) ;  add_164 = None
        mm_146 = torch.mm(view_523,t_176) ;  view_523 = t_176 = None
        view_524 = mm_146.view(1, 5, 512, 2304) ;  mm_146 = None
        view_525 = view_524.view(1, 5, 512, 16, 144) ;  view_524 = None
        permute_78 = view_525.permute(0, 3, 1, 2, 4) ;  view_525 = None
        split_tensor_78 = torch.split(permute_78,48,dim = -1) ;  permute_78 = None
        getitem_389 = split_tensor_78[0]
        getitem_390 = split_tensor_78[1]
        getitem_391 = split_tensor_78[2];  split_tensor_78 = None
        view_526 = arg287_1.view(1, 16, 1, 1, 48) ;  arg287_1 = None
        add_165 = torch.add(getitem_389,view_526) ;  getitem_389 = view_526 = None
        view_527 = bitwise_and_1.view(1, 1, 512, 512, 1) 
        bitwise_not_36 = torch.bitwise_not(view_527) ;  view_527 = None
        masked_fill_36 = view_520.masked_fill(bitwise_not_36,-10000) ;  view_520 = bitwise_not_36 = None
        permute_79 = masked_fill_36.permute(0, 4, 1, 2, 3) ;  masked_fill_36 = None
        view_528 = bitwise_and_1.view(1, 1, 1, 512, 512) ;  bitwise_and_1 = None
        bitwise_not_37 = torch.bitwise_not(view_528) ;  view_528 = None
        masked_fill_37 = permute_79.masked_fill(bitwise_not_37,-10000) ;  permute_79 = bitwise_not_37 = None
        mul_130 = torch.mul(add_165,0.3799178428257963) ;  add_165 = None
        transpose_15 = torch.transpose(getitem_390,-2,-1) ;  getitem_390 = None
        mul_131 = torch.mul(transpose_15,0.3799178428257963) ;  transpose_15 = None
        expand_74 = mul_130.expand(1, 16, 5, 512, 48) ;  mul_130 = None
        clone_82 = torch.clone(expand_74,memory_format = torch.contiguous_format) ;  expand_74 = None
        _unsafe_view_81 = clone_82.view(80, 512, 48) ;  clone_82 = None
        expand_75 = mul_131.expand(1, 16, 5, 48, 512) ;  mul_131 = None
        clone_83 = torch.clone(expand_75,memory_format = torch.contiguous_format) ;  expand_75 = None
        _unsafe_view_82 = clone_83.view(80, 48, 512) ;  clone_83 = None
        bmm_36 = torch.bmm(_unsafe_view_81,_unsafe_view_82) ;  _unsafe_view_81 = _unsafe_view_82 = None
        view_529 = bmm_36.view(1, 16, 5, 512, 512) ;  bmm_36 = None
        add_166 = torch.add(view_529,masked_fill_37) ;  view_529 = masked_fill_37 = None
        _softmax_15 = torch.softmax(add_166, -1)
        expand_76 = _softmax_15.expand(1, 16, 5, 512, 512) ;  _softmax_15 = None
        view_530 = expand_76.view(80, 512, 512) ;  expand_76 = None
        expand_77 = getitem_391.expand(1, 16, 5, 512, 48) ;  getitem_391 = None
        clone_84 = torch.clone(expand_77,memory_format = torch.contiguous_format) ;  expand_77 = None
        _unsafe_view_83 = clone_84.view(80, 512, 48) ;  clone_84 = None
        bmm_37 = torch.bmm(view_530,_unsafe_view_83) ;  view_530 = _unsafe_view_83 = None
        view_531 = bmm_37.view(1, 16, 5, 512, 48) ;  bmm_37 = None
        permute_80 = view_531.permute(0, 2, 3, 1, 4) ;  view_531 = None
        clone_85 = torch.clone(permute_80,memory_format = torch.contiguous_format) ;  permute_80 = None
        _unsafe_view_84 = clone_85.view(1, 5, 512, 768) ;  clone_85 = None
        t_177 = arg300_1.t() ;  arg300_1 = None
        view_532 = _unsafe_view_84.view(2560, 768) ;  _unsafe_view_84 = None
        mm_147 = torch.mm(view_532,t_177) ;  view_532 = t_177 = None
        view_533 = mm_147.view(1, 5, 512, 768) ;  mm_147 = None
        view_534 = getitem_33.view(2560, 384) 
        t_178 = arg295_1.t() ;  arg295_1 = None
        addmm_30 = torch.addmm(arg296_1,view_534,t_178) ;  arg296_1 = view_534 = t_178 = None
        view_535 = addmm_30.view(1, 5, 512, 768) ;  addmm_30 = None
        sigmoid_36 = torch.sigmoid(view_535) ;  view_535 = None
        mul_132 = torch.mul(sigmoid_36,view_533) ;  sigmoid_36 = view_533 = None
        native_layer_norm_default_66 = (torch.nn.functional.layer_norm(add_162,[768],None,None,0.1),) 
        getitem_392 = native_layer_norm_default_66[0]
        t_179 = arg288_1.t() ;  arg288_1 = None
        view_536 = getitem_33.view(2560, 384) 
        mm_148 = torch.mm(view_536,t_179) ;  view_536 = t_179 = None
        view_537 = mm_148.view(1, 5, 512, 1536) ;  mm_148 = None
        split_tensor_79 = torch.split(view_537,768,dim = -1) ;  view_537 = None
        getitem_395 = split_tensor_79[0]
        getitem_396 = split_tensor_79[1];  split_tensor_79 = None
        add_167 = torch.add(getitem_395,1) ;  getitem_395 = None
        mul_133 = torch.mul(getitem_392,add_167) ;  getitem_392 = add_167 = None
        add_168 = torch.add(mul_133,getitem_396) ;  mul_133 = getitem_396 = None
        t_180 = arg289_1.t() ;  arg289_1 = None
        view_538 = add_168.view(2560, 768) ;  add_168 = None
        mm_149 = torch.mm(view_538,t_180) ;  view_538 = t_180 = None
        view_539 = mm_149.view(1, 5, 512, 3072) ;  mm_149 = None
        split_tensor_80 = torch.split(view_539,1536,dim = -1) ;  view_539 = None
        getitem_397 = split_tensor_80[0]
        getitem_398 = split_tensor_80[1];  split_tensor_80 = None
        silu_22 = torch.nn.functional.silu(getitem_397) ;  getitem_397 = None
        mul_134 = torch.mul(silu_22,getitem_398) ;  silu_22 = getitem_398 = None
        view_540 = getitem_33.view(2560, 384) ;  getitem_33 = None
        t_181 = arg291_1.t() ;  arg291_1 = None
        addmm_31 = torch.addmm(arg292_1,view_540,t_181) ;  arg292_1 = view_540 = t_181 = None
        view_541 = addmm_31.view(1, 5, 512, 768) ;  addmm_31 = None
        sigmoid_37 = torch.sigmoid(view_541) ;  view_541 = None
        t_182 = arg290_1.t() ;  arg290_1 = None
        view_542 = mul_134.view(2560, 1536) ;  mul_134 = None
        mm_150 = torch.mm(view_542,t_182) ;  view_542 = t_182 = None
        view_543 = mm_150.view(1, 5, 512, 768) ;  mm_150 = None
        mul_135 = torch.mul(sigmoid_37,view_543) ;  sigmoid_37 = view_543 = None
        add_169 = torch.add(mul_132,mul_135) ;  mul_132 = mul_135 = None
        add_170 = torch.add(add_162,add_169) ;  add_162 = add_169 = None
        native_layer_norm_default_67 = (torch.nn.functional.layer_norm(add_170,[768],arg339_1,arg340_1,1e-05),) ;  add_170 = arg339_1 = arg340_1 = None
        getitem_399 = native_layer_norm_default_67[0]
        native_layer_norm_default_68 = (torch.nn.functional.layer_norm(getitem_42,[128],arg341_1,arg342_1,1e-05),) ;  getitem_42 = arg341_1 = arg342_1 = None
        getitem_402 = native_layer_norm_default_68[0]
        view_544 = getitem_399.view(5, 512, 768) ;  getitem_399 = None
        view_545 = view_127.view(5, 11776, 128) ;  view_127 = None
        unsqueeze_48 = torch.unsqueeze(getitem_402,1) ;  getitem_402 = None
        expand_78 = unsqueeze_48.expand(-1, 5, -1, -1) ;  unsqueeze_48 = None
        view_546 = expand_78.view(5, 11776, 128) ;  expand_78 = None
        unsqueeze_49 = torch.unsqueeze(arg350_1,1) ;  arg350_1 = None
        expand_79 = unsqueeze_49.expand(-1, 5, -1, -1, -1) ;  unsqueeze_49 = None
        view_548 = expand_79.view(5, 368, 32, 128) ;  expand_79 = None
        unsqueeze_50 = torch.unsqueeze(arg356_1,1) ;  arg356_1 = None
        expand_80 = unsqueeze_50.expand(-1, 5, -1) ;  unsqueeze_50 = None
        view_549 = expand_80.view(5, 11776) ;  expand_80 = None
        unsqueeze_51 = torch.unsqueeze(arg349_1,1) ;  arg349_1 = None
        expand_81 = unsqueeze_51.expand(-1, 5, -1) ;  unsqueeze_51 = None
        view_550 = expand_81.view(5, 11776) ;  expand_81 = None
        t_183 = arg301_1.t() ;  arg301_1 = None
        view_551 = view_544.view(2560, 768) ;  view_544 = None
        mm_151 = torch.mm(view_551,t_183) ;  view_551 = t_183 = None
        view_552 = mm_151.view(5, 512, 128) ;  mm_151 = None
        arange_4 = torch.arange(5,device = self.device,pin_memory = False) 
        unsqueeze_52 = torch.unsqueeze(arange_4,1) ;  arange_4 = None
        index_12 = view_552[unsqueeze_52,view_549] ;  view_552 = unsqueeze_52 = view_549 = None
        clone_86 = torch.clone(index_12) ;  index_12 = None
        add_171 = torch.add(view_545,clone_86) ;  view_545 = clone_86 = None
        arange_5 = torch.arange(11776,device = self.device,pin_memory = False) 
        view_553 = arange_5.view(368, 32) ;  arange_5 = None
        slice_21 = view_553[0:] ;  view_553 = None
        slice_22 = slice_21[:, 0:1] ;  slice_21 = None
        add_172 = torch.add(slice_22,-48) ;  slice_22 = None
        arange_6 = torch.arange(128,device = self.device,pin_memory = False) 
        add_173 = torch.add(add_172,arange_6) ;  add_172 = arange_6 = None
        remainder_1 = torch.remainder(add_173,11776) ;  add_173 = None
        view_554 = view_57.view(1, 5, 368, 32, 128, 16) ;  view_57 = None
        view_555 = view_554.view(5, 368, 32, 128, 16) ;  view_554 = None
        native_layer_norm_default_69 = (torch.nn.functional.layer_norm(view_555,[16],arg332_1,arg333_1,1e-05),) ;  view_555 = arg332_1 = arg333_1 = None
        getitem_405 = native_layer_norm_default_69[0]
        unbind_int_4 = torch.unbind(arg334_1) ;  arg334_1 = None
        getitem_408 = unbind_int_4[0]
        getitem_409 = unbind_int_4[1]
        getitem_410 = unbind_int_4[2];  unbind_int_4 = None
        unsqueeze_53 = torch.unsqueeze(view_550,-1) 
        bitwise_not_38 = torch.bitwise_not(unsqueeze_53) ;  unsqueeze_53 = None
        masked_fill_38 = add_171.masked_fill(bitwise_not_38,0.0) ;  add_171 = bitwise_not_38 = None
        unsqueeze_54 = torch.unsqueeze(getitem_405,5) 
        permute_81 = unsqueeze_54.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_54 = None
        unsqueeze_55 = torch.unsqueeze(getitem_408,2) ;  getitem_408 = None
        unsqueeze_56 = torch.unsqueeze(unsqueeze_55,3) ;  unsqueeze_55 = None
        unsqueeze_57 = torch.unsqueeze(unsqueeze_56,4) ;  unsqueeze_56 = None
        unsqueeze_58 = torch.unsqueeze(unsqueeze_57,5) ;  unsqueeze_57 = None
        permute_82 = unsqueeze_58.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_58 = None
        permute_83 = permute_81.permute(0, 2, 3, 4, 5, 1) ;  permute_81 = None
        view_556 = permute_83.view(1, 7536640, 16) ;  permute_83 = None
        permute_84 = permute_82.permute(5, 1, 0, 2, 3, 4) ;  permute_82 = None
        view_557 = permute_84.view(1, 16, 4) ;  permute_84 = None
        bmm_38 = torch.bmm(view_556,view_557) ;  view_556 = view_557 = None
        view_558 = bmm_38.view(5, 368, 32, 128, 1, 4) ;  bmm_38 = None
        permute_85 = view_558.permute(0, 5, 1, 2, 3, 4) ;  view_558 = None
        view_559 = permute_85.view(5, 4, 368, 32, 128) ;  permute_85 = None
        view_560 = view_548.view(5, 1, 368, 32, 128) 
        bitwise_not_39 = torch.bitwise_not(view_560) ;  view_560 = None
        masked_fill_39 = view_559.masked_fill(bitwise_not_39,-10000) ;  view_559 = bitwise_not_39 = None
        native_layer_norm_default_70 = (torch.nn.functional.layer_norm(masked_fill_38,[128],None,None,0.1),) 
        getitem_411 = native_layer_norm_default_70[0]
        t_184 = arg318_1.t() ;  arg318_1 = None
        clone_87 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_85 = clone_87.view(58880, 128) ;  clone_87 = None
        mm_152 = torch.mm(_unsafe_view_85,t_184) ;  _unsafe_view_85 = t_184 = None
        view_561 = mm_152.view(5, 11776, 256) ;  mm_152 = None
        split_tensor_81 = torch.split(view_561,128,dim = -1) ;  view_561 = None
        getitem_414 = split_tensor_81[0]
        getitem_415 = split_tensor_81[1];  split_tensor_81 = None
        add_174 = torch.add(getitem_414,1) ;  getitem_414 = None
        mul_136 = torch.mul(getitem_411,add_174) ;  getitem_411 = add_174 = None
        add_175 = torch.add(mul_136,getitem_415) ;  mul_136 = getitem_415 = None
        unsqueeze_59 = torch.unsqueeze(add_175,3) ;  add_175 = None
        unsqueeze_60 = torch.unsqueeze(unsqueeze_59,4) ;  unsqueeze_59 = None
        unsqueeze_61 = torch.unsqueeze(unsqueeze_60,5) ;  unsqueeze_60 = None
        permute_86 = unsqueeze_61.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_61 = None
        unsqueeze_62 = torch.unsqueeze(arg319_1,4) ;  arg319_1 = None
        unsqueeze_63 = torch.unsqueeze(unsqueeze_62,5) ;  unsqueeze_62 = None
        permute_87 = unsqueeze_63.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_63 = None
        permute_88 = permute_86.permute(1, 3, 5, 0, 2, 4) ;  permute_86 = None
        view_562 = permute_88.view(1, 58880, 128) ;  permute_88 = None
        permute_89 = permute_87.permute(5, 0, 2, 4, 1, 3) ;  permute_87 = None
        view_563 = permute_89.view(1, 128, 384) ;  permute_89 = None
        bmm_39 = torch.bmm(view_562,view_563) ;  view_562 = view_563 = None
        view_564 = bmm_39.view(5, 11776, 1, 3, 4, 32) ;  bmm_39 = None
        permute_90 = view_564.permute(3, 0, 4, 1, 5, 2) ;  view_564 = None
        view_565 = permute_90.view(3, 5, 4, 11776, 32) ;  permute_90 = None
        clone_88 = torch.clone(view_565,memory_format = torch.contiguous_format) ;  view_565 = None
        _unsafe_view_86 = clone_88.view(3, 20, 11776, 32) ;  clone_88 = None
        unbind_int_5 = torch.unbind(_unsafe_view_86) ;  _unsafe_view_86 = None
        getitem_416 = unbind_int_5[0]
        getitem_417 = unbind_int_5[1]
        getitem_418 = unbind_int_5[2];  unbind_int_5 = None
        unsqueeze_64 = torch.unsqueeze(arg317_1,0) ;  arg317_1 = None
        expand_82 = unsqueeze_64.expand(5, -1, -1) ;  unsqueeze_64 = None
        clone_89 = torch.clone(expand_82,memory_format = torch.contiguous_format) ;  expand_82 = None
        _unsafe_view_87 = clone_89.view(20, 1, 32) ;  clone_89 = None
        add_176 = torch.add(getitem_416,_unsafe_view_87) ;  getitem_416 = _unsafe_view_87 = None
        view_566 = add_176.view(20, 368, 32, 32) ;  add_176 = None
        slice_23 = getitem_417[0:] ;  getitem_417 = None
        slice_24 = slice_23[:, :, 0:] ;  slice_23 = None
        index_13 = slice_24[:,remainder_1] ;  slice_24 = None
        slice_25 = getitem_418[0:] ;  getitem_418 = None
        slice_26 = slice_25[:, :, 0:] ;  slice_25 = None
        index_14 = slice_26[:,remainder_1] ;  slice_26 = None
        view_567 = masked_fill_39.view(20, 368, 32, 128) ;  masked_fill_39 = None
        expand_83 = view_567.expand(20, 368, 32, 128) ;  view_567 = None
        _scaled_dot_product_efficient_attention_default_3 = (torch.nn.functional.scaled_dot_product_attention(view_566,index_13,index_14,expand_83,False),) ;  view_566 = index_13 = index_14 = expand_83 = None
        getitem_419 = _scaled_dot_product_efficient_attention_default_3[0]
        view_568 = getitem_419.view(5, 4, 368, 32, 32) ;  getitem_419 = None
        permute_91 = view_568.permute(0, 2, 3, 1, 4) ;  view_568 = None
        clone_90 = torch.clone(permute_91,memory_format = torch.contiguous_format) ;  permute_91 = None
        _unsafe_view_88 = clone_90.view(5, 11776, 128) ;  clone_90 = None
        t_185 = arg320_1.t() ;  arg320_1 = None
        clone_91 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_89 = clone_91.view(58880, 128) ;  clone_91 = None
        mm_153 = torch.mm(_unsafe_view_89,t_185) ;  _unsafe_view_89 = t_185 = None
        view_569 = mm_153.view(5, 11776, 128) ;  mm_153 = None
        add_177 = torch.add(view_569,arg321_1) ;  view_569 = arg321_1 = None
        sigmoid_38 = torch.sigmoid(add_177) ;  add_177 = None
        mul_137 = torch.mul(_unsafe_view_88,sigmoid_38) ;  _unsafe_view_88 = sigmoid_38 = None
        native_layer_norm_default_71 = (torch.nn.functional.layer_norm(masked_fill_38,[128],None,None,0.1),) 
        getitem_423 = native_layer_norm_default_71[0]
        t_186 = arg302_1.t() ;  arg302_1 = None
        clone_92 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_90 = clone_92.view(58880, 128) ;  clone_92 = None
        mm_154 = torch.mm(_unsafe_view_90,t_186) ;  _unsafe_view_90 = t_186 = None
        view_570 = mm_154.view(5, 11776, 256) ;  mm_154 = None
        split_tensor_82 = torch.split(view_570,128,dim = -1) ;  view_570 = None
        getitem_426 = split_tensor_82[0]
        getitem_427 = split_tensor_82[1];  split_tensor_82 = None
        add_178 = torch.add(getitem_426,1) ;  getitem_426 = None
        mul_138 = torch.mul(getitem_423,add_178) ;  getitem_423 = add_178 = None
        add_179 = torch.add(mul_138,getitem_427) ;  mul_138 = getitem_427 = None
        t_187 = arg303_1.t() ;  arg303_1 = None
        view_571 = add_179.view(58880, 128) ;  add_179 = None
        mm_155 = torch.mm(view_571,t_187) ;  view_571 = t_187 = None
        view_572 = mm_155.view(5, 11776, 512) ;  mm_155 = None
        split_tensor_83 = torch.split(view_572,256,dim = -1) ;  view_572 = None
        getitem_428 = split_tensor_83[0]
        getitem_429 = split_tensor_83[1];  split_tensor_83 = None
        silu_23 = torch.nn.functional.silu(getitem_428) ;  getitem_428 = None
        mul_139 = torch.mul(silu_23,getitem_429) ;  silu_23 = getitem_429 = None
        t_188 = arg305_1.t() ;  arg305_1 = None
        clone_93 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_91 = clone_93.view(58880, 128) ;  clone_93 = None
        mm_156 = torch.mm(_unsafe_view_91,t_188) ;  _unsafe_view_91 = t_188 = None
        view_573 = mm_156.view(5, 11776, 128) ;  mm_156 = None
        add_180 = torch.add(view_573,arg306_1) ;  view_573 = arg306_1 = None
        sigmoid_39 = torch.sigmoid(add_180) ;  add_180 = None
        t_189 = arg304_1.t() ;  arg304_1 = None
        view_574 = mul_139.view(58880, 256) ;  mul_139 = None
        mm_157 = torch.mm(view_574,t_189) ;  view_574 = t_189 = None
        view_575 = mm_157.view(5, 11776, 128) ;  mm_157 = None
        mul_140 = torch.mul(sigmoid_39,view_575) ;  sigmoid_39 = view_575 = None
        add_181 = torch.add(masked_fill_38,mul_140) ;  masked_fill_38 = mul_140 = None
        add_182 = torch.add(add_181,mul_137) ;  add_181 = mul_137 = None
        unsqueeze_65 = torch.unsqueeze(view_550,-1) 
        bitwise_not_40 = torch.bitwise_not(unsqueeze_65) ;  unsqueeze_65 = None
        masked_fill_40 = add_182.masked_fill(bitwise_not_40,0.0) ;  add_182 = bitwise_not_40 = None
        unsqueeze_66 = torch.unsqueeze(getitem_405,5) 
        permute_92 = unsqueeze_66.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_66 = None
        unsqueeze_67 = torch.unsqueeze(getitem_409,2) ;  getitem_409 = None
        unsqueeze_68 = torch.unsqueeze(unsqueeze_67,3) ;  unsqueeze_67 = None
        unsqueeze_69 = torch.unsqueeze(unsqueeze_68,4) ;  unsqueeze_68 = None
        unsqueeze_70 = torch.unsqueeze(unsqueeze_69,5) ;  unsqueeze_69 = None
        permute_93 = unsqueeze_70.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_70 = None
        permute_94 = permute_92.permute(0, 2, 3, 4, 5, 1) ;  permute_92 = None
        view_576 = permute_94.view(1, 7536640, 16) ;  permute_94 = None
        permute_95 = permute_93.permute(5, 1, 0, 2, 3, 4) ;  permute_93 = None
        view_577 = permute_95.view(1, 16, 4) ;  permute_95 = None
        bmm_40 = torch.bmm(view_576,view_577) ;  view_576 = view_577 = None
        view_578 = bmm_40.view(5, 368, 32, 128, 1, 4) ;  bmm_40 = None
        permute_96 = view_578.permute(0, 5, 1, 2, 3, 4) ;  view_578 = None
        view_579 = permute_96.view(5, 4, 368, 32, 128) ;  permute_96 = None
        view_580 = view_548.view(5, 1, 368, 32, 128) 
        bitwise_not_41 = torch.bitwise_not(view_580) ;  view_580 = None
        masked_fill_41 = view_579.masked_fill(bitwise_not_41,-10000) ;  view_579 = bitwise_not_41 = None
        native_layer_norm_default_72 = (torch.nn.functional.layer_norm(masked_fill_40,[128],None,None,0.1),) 
        getitem_430 = native_layer_norm_default_72[0]
        t_190 = arg323_1.t() ;  arg323_1 = None
        clone_94 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_92 = clone_94.view(58880, 128) ;  clone_94 = None
        mm_158 = torch.mm(_unsafe_view_92,t_190) ;  _unsafe_view_92 = t_190 = None
        view_581 = mm_158.view(5, 11776, 256) ;  mm_158 = None
        split_tensor_84 = torch.split(view_581,128,dim = -1) ;  view_581 = None
        getitem_433 = split_tensor_84[0]
        getitem_434 = split_tensor_84[1];  split_tensor_84 = None
        add_183 = torch.add(getitem_433,1) ;  getitem_433 = None
        mul_141 = torch.mul(getitem_430,add_183) ;  getitem_430 = add_183 = None
        add_184 = torch.add(mul_141,getitem_434) ;  mul_141 = getitem_434 = None
        unsqueeze_71 = torch.unsqueeze(add_184,3) ;  add_184 = None
        unsqueeze_72 = torch.unsqueeze(unsqueeze_71,4) ;  unsqueeze_71 = None
        unsqueeze_73 = torch.unsqueeze(unsqueeze_72,5) ;  unsqueeze_72 = None
        permute_97 = unsqueeze_73.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_73 = None
        unsqueeze_74 = torch.unsqueeze(arg324_1,4) ;  arg324_1 = None
        unsqueeze_75 = torch.unsqueeze(unsqueeze_74,5) ;  unsqueeze_74 = None
        permute_98 = unsqueeze_75.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_75 = None
        permute_99 = permute_97.permute(1, 3, 5, 0, 2, 4) ;  permute_97 = None
        view_582 = permute_99.view(1, 58880, 128) ;  permute_99 = None
        permute_100 = permute_98.permute(5, 0, 2, 4, 1, 3) ;  permute_98 = None
        view_583 = permute_100.view(1, 128, 384) ;  permute_100 = None
        bmm_41 = torch.bmm(view_582,view_583) ;  view_582 = view_583 = None
        view_584 = bmm_41.view(5, 11776, 1, 3, 4, 32) ;  bmm_41 = None
        permute_101 = view_584.permute(3, 0, 4, 1, 5, 2) ;  view_584 = None
        view_585 = permute_101.view(3, 5, 4, 11776, 32) ;  permute_101 = None
        clone_95 = torch.clone(view_585,memory_format = torch.contiguous_format) ;  view_585 = None
        _unsafe_view_93 = clone_95.view(3, 20, 11776, 32) ;  clone_95 = None
        unbind_int_6 = torch.unbind(_unsafe_view_93) ;  _unsafe_view_93 = None
        getitem_435 = unbind_int_6[0]
        getitem_436 = unbind_int_6[1]
        getitem_437 = unbind_int_6[2];  unbind_int_6 = None
        unsqueeze_76 = torch.unsqueeze(arg322_1,0) ;  arg322_1 = None
        expand_84 = unsqueeze_76.expand(5, -1, -1) ;  unsqueeze_76 = None
        clone_96 = torch.clone(expand_84,memory_format = torch.contiguous_format) ;  expand_84 = None
        _unsafe_view_94 = clone_96.view(20, 1, 32) ;  clone_96 = None
        add_185 = torch.add(getitem_435,_unsafe_view_94) ;  getitem_435 = _unsafe_view_94 = None
        view_586 = add_185.view(20, 368, 32, 32) ;  add_185 = None
        slice_27 = getitem_436[0:] ;  getitem_436 = None
        slice_28 = slice_27[:, :, 0:] ;  slice_27 = None
        index_15 = slice_28[:,remainder_1] ;  slice_28 = None
        slice_29 = getitem_437[0:] ;  getitem_437 = None
        slice_30 = slice_29[:, :, 0:] ;  slice_29 = None
        index_16 = slice_30[:,remainder_1] ;  slice_30 = None
        view_587 = masked_fill_41.view(20, 368, 32, 128) ;  masked_fill_41 = None
        expand_85 = view_587.expand(20, 368, 32, 128) ;  view_587 = None
        _scaled_dot_product_efficient_attention_default_4 = (torch.nn.functional.scaled_dot_product_attention(view_586,index_15,index_16,expand_85,False),) ;  view_586 = index_15 = index_16 = expand_85 = None
        getitem_438 = _scaled_dot_product_efficient_attention_default_4[0]
        view_588 = getitem_438.view(5, 4, 368, 32, 32) ;  getitem_438 = None
        permute_102 = view_588.permute(0, 2, 3, 1, 4) ;  view_588 = None
        clone_97 = torch.clone(permute_102,memory_format = torch.contiguous_format) ;  permute_102 = None
        _unsafe_view_95 = clone_97.view(5, 11776, 128) ;  clone_97 = None
        t_191 = arg325_1.t() ;  arg325_1 = None
        clone_98 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_96 = clone_98.view(58880, 128) ;  clone_98 = None
        mm_159 = torch.mm(_unsafe_view_96,t_191) ;  _unsafe_view_96 = t_191 = None
        view_589 = mm_159.view(5, 11776, 128) ;  mm_159 = None
        add_186 = torch.add(view_589,arg326_1) ;  view_589 = arg326_1 = None
        sigmoid_40 = torch.sigmoid(add_186) ;  add_186 = None
        mul_142 = torch.mul(_unsafe_view_95,sigmoid_40) ;  _unsafe_view_95 = sigmoid_40 = None
        native_layer_norm_default_73 = (torch.nn.functional.layer_norm(masked_fill_40,[128],None,None,0.1),) 
        getitem_442 = native_layer_norm_default_73[0]
        t_192 = arg307_1.t() ;  arg307_1 = None
        clone_99 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_97 = clone_99.view(58880, 128) ;  clone_99 = None
        mm_160 = torch.mm(_unsafe_view_97,t_192) ;  _unsafe_view_97 = t_192 = None
        view_590 = mm_160.view(5, 11776, 256) ;  mm_160 = None
        split_tensor_85 = torch.split(view_590,128,dim = -1) ;  view_590 = None
        getitem_445 = split_tensor_85[0]
        getitem_446 = split_tensor_85[1];  split_tensor_85 = None
        add_187 = torch.add(getitem_445,1) ;  getitem_445 = None
        mul_143 = torch.mul(getitem_442,add_187) ;  getitem_442 = add_187 = None
        add_188 = torch.add(mul_143,getitem_446) ;  mul_143 = getitem_446 = None
        t_193 = arg308_1.t() ;  arg308_1 = None
        view_591 = add_188.view(58880, 128) ;  add_188 = None
        mm_161 = torch.mm(view_591,t_193) ;  view_591 = t_193 = None
        view_592 = mm_161.view(5, 11776, 512) ;  mm_161 = None
        split_tensor_86 = torch.split(view_592,256,dim = -1) ;  view_592 = None
        getitem_447 = split_tensor_86[0]
        getitem_448 = split_tensor_86[1];  split_tensor_86 = None
        silu_24 = torch.nn.functional.silu(getitem_447) ;  getitem_447 = None
        mul_144 = torch.mul(silu_24,getitem_448) ;  silu_24 = getitem_448 = None
        t_194 = arg310_1.t() ;  arg310_1 = None
        clone_100 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_98 = clone_100.view(58880, 128) ;  clone_100 = None
        mm_162 = torch.mm(_unsafe_view_98,t_194) ;  _unsafe_view_98 = t_194 = None
        view_593 = mm_162.view(5, 11776, 128) ;  mm_162 = None
        add_189 = torch.add(view_593,arg311_1) ;  view_593 = arg311_1 = None
        sigmoid_41 = torch.sigmoid(add_189) ;  add_189 = None
        t_195 = arg309_1.t() ;  arg309_1 = None
        view_594 = mul_144.view(58880, 256) ;  mul_144 = None
        mm_163 = torch.mm(view_594,t_195) ;  view_594 = t_195 = None
        view_595 = mm_163.view(5, 11776, 128) ;  mm_163 = None
        mul_145 = torch.mul(sigmoid_41,view_595) ;  sigmoid_41 = view_595 = None
        add_190 = torch.add(masked_fill_40,mul_145) ;  masked_fill_40 = mul_145 = None
        add_191 = torch.add(add_190,mul_142) ;  add_190 = mul_142 = None
        unsqueeze_77 = torch.unsqueeze(view_550,-1) ;  view_550 = None
        bitwise_not_42 = torch.bitwise_not(unsqueeze_77) ;  unsqueeze_77 = None
        masked_fill_42 = add_191.masked_fill(bitwise_not_42,0.0) ;  add_191 = bitwise_not_42 = None
        unsqueeze_78 = torch.unsqueeze(getitem_405,5) ;  getitem_405 = None
        permute_103 = unsqueeze_78.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_78 = None
        unsqueeze_79 = torch.unsqueeze(getitem_410,2) ;  getitem_410 = None
        unsqueeze_80 = torch.unsqueeze(unsqueeze_79,3) ;  unsqueeze_79 = None
        unsqueeze_81 = torch.unsqueeze(unsqueeze_80,4) ;  unsqueeze_80 = None
        unsqueeze_82 = torch.unsqueeze(unsqueeze_81,5) ;  unsqueeze_81 = None
        permute_104 = unsqueeze_82.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_82 = None
        permute_105 = permute_103.permute(0, 2, 3, 4, 5, 1) ;  permute_103 = None
        view_596 = permute_105.view(1, 7536640, 16) ;  permute_105 = None
        permute_106 = permute_104.permute(5, 1, 0, 2, 3, 4) ;  permute_104 = None
        view_597 = permute_106.view(1, 16, 4) ;  permute_106 = None
        bmm_42 = torch.bmm(view_596,view_597) ;  view_596 = view_597 = None
        view_598 = bmm_42.view(5, 368, 32, 128, 1, 4) ;  bmm_42 = None
        permute_107 = view_598.permute(0, 5, 1, 2, 3, 4) ;  view_598 = None
        view_599 = permute_107.view(5, 4, 368, 32, 128) ;  permute_107 = None
        view_600 = view_548.view(5, 1, 368, 32, 128) ;  view_548 = None
        bitwise_not_43 = torch.bitwise_not(view_600) ;  view_600 = None
        masked_fill_43 = view_599.masked_fill(bitwise_not_43,-10000) ;  view_599 = bitwise_not_43 = None
        native_layer_norm_default_74 = (torch.nn.functional.layer_norm(masked_fill_42,[128],None,None,0.1),) 
        getitem_449 = native_layer_norm_default_74[0]
        t_196 = arg328_1.t() ;  arg328_1 = None
        clone_101 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_99 = clone_101.view(58880, 128) ;  clone_101 = None
        mm_164 = torch.mm(_unsafe_view_99,t_196) ;  _unsafe_view_99 = t_196 = None
        view_601 = mm_164.view(5, 11776, 256) ;  mm_164 = None
        split_tensor_87 = torch.split(view_601,128,dim = -1) ;  view_601 = None
        getitem_452 = split_tensor_87[0]
        getitem_453 = split_tensor_87[1];  split_tensor_87 = None
        add_192 = torch.add(getitem_452,1) ;  getitem_452 = None
        mul_146 = torch.mul(getitem_449,add_192) ;  getitem_449 = add_192 = None
        add_193 = torch.add(mul_146,getitem_453) ;  mul_146 = getitem_453 = None
        unsqueeze_83 = torch.unsqueeze(add_193,3) ;  add_193 = None
        unsqueeze_84 = torch.unsqueeze(unsqueeze_83,4) ;  unsqueeze_83 = None
        unsqueeze_85 = torch.unsqueeze(unsqueeze_84,5) ;  unsqueeze_84 = None
        permute_108 = unsqueeze_85.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_85 = None
        unsqueeze_86 = torch.unsqueeze(arg329_1,4) ;  arg329_1 = None
        unsqueeze_87 = torch.unsqueeze(unsqueeze_86,5) ;  unsqueeze_86 = None
        permute_109 = unsqueeze_87.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_87 = None
        permute_110 = permute_108.permute(1, 3, 5, 0, 2, 4) ;  permute_108 = None
        view_602 = permute_110.view(1, 58880, 128) ;  permute_110 = None
        permute_111 = permute_109.permute(5, 0, 2, 4, 1, 3) ;  permute_109 = None
        view_603 = permute_111.view(1, 128, 384) ;  permute_111 = None
        bmm_43 = torch.bmm(view_602,view_603) ;  view_602 = view_603 = None
        view_604 = bmm_43.view(5, 11776, 1, 3, 4, 32) ;  bmm_43 = None
        permute_112 = view_604.permute(3, 0, 4, 1, 5, 2) ;  view_604 = None
        view_605 = permute_112.view(3, 5, 4, 11776, 32) ;  permute_112 = None
        clone_102 = torch.clone(view_605,memory_format = torch.contiguous_format) ;  view_605 = None
        _unsafe_view_100 = clone_102.view(3, 20, 11776, 32) ;  clone_102 = None
        unbind_int_7 = torch.unbind(_unsafe_view_100) ;  _unsafe_view_100 = None
        getitem_454 = unbind_int_7[0]
        getitem_455 = unbind_int_7[1]
        getitem_456 = unbind_int_7[2];  unbind_int_7 = None
        unsqueeze_88 = torch.unsqueeze(arg327_1,0) ;  arg327_1 = None
        expand_86 = unsqueeze_88.expand(5, -1, -1) ;  unsqueeze_88 = None
        clone_103 = torch.clone(expand_86,memory_format = torch.contiguous_format) ;  expand_86 = None
        _unsafe_view_101 = clone_103.view(20, 1, 32) ;  clone_103 = None
        add_194 = torch.add(getitem_454,_unsafe_view_101) ;  getitem_454 = _unsafe_view_101 = None
        view_606 = add_194.view(20, 368, 32, 32) ;  add_194 = None
        slice_31 = getitem_455[0:] ;  getitem_455 = None
        slice_32 = slice_31[:, :, 0:] ;  slice_31 = None
        index_17 = slice_32[:,remainder_1] ;  slice_32 = None
        slice_33 = getitem_456[0:] ;  getitem_456 = None
        slice_34 = slice_33[:, :, 0:] ;  slice_33 = None
        index_18 = slice_34[:,remainder_1] ;  slice_34 = remainder_1 = None
        view_607 = masked_fill_43.view(20, 368, 32, 128) ;  masked_fill_43 = None
        expand_87 = view_607.expand(20, 368, 32, 128) ;  view_607 = None
        _scaled_dot_product_efficient_attention_default_5 = (torch.nn.functional.scaled_dot_product_attention(view_606,index_17,index_18,expand_87,False),) ;  view_606 = index_17 = index_18 = expand_87 = None
        getitem_457 = _scaled_dot_product_efficient_attention_default_5[0]
        view_608 = getitem_457.view(5, 4, 368, 32, 32) ;  getitem_457 = None
        permute_113 = view_608.permute(0, 2, 3, 1, 4) ;  view_608 = None
        clone_104 = torch.clone(permute_113,memory_format = torch.contiguous_format) ;  permute_113 = None
        _unsafe_view_102 = clone_104.view(5, 11776, 128) ;  clone_104 = None
        t_197 = arg330_1.t() ;  arg330_1 = None
        clone_105 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_103 = clone_105.view(58880, 128) ;  clone_105 = None
        mm_165 = torch.mm(_unsafe_view_103,t_197) ;  _unsafe_view_103 = t_197 = None
        view_609 = mm_165.view(5, 11776, 128) ;  mm_165 = None
        add_195 = torch.add(view_609,arg331_1) ;  view_609 = arg331_1 = None
        sigmoid_42 = torch.sigmoid(add_195) ;  add_195 = None
        mul_147 = torch.mul(_unsafe_view_102,sigmoid_42) ;  _unsafe_view_102 = sigmoid_42 = None
        native_layer_norm_default_75 = (torch.nn.functional.layer_norm(masked_fill_42,[128],None,None,0.1),) 
        getitem_461 = native_layer_norm_default_75[0]
        t_198 = arg312_1.t() ;  arg312_1 = None
        clone_106 = torch.clone(view_546,memory_format = torch.contiguous_format) 
        _unsafe_view_104 = clone_106.view(58880, 128) ;  clone_106 = None
        mm_166 = torch.mm(_unsafe_view_104,t_198) ;  _unsafe_view_104 = t_198 = None
        view_610 = mm_166.view(5, 11776, 256) ;  mm_166 = None
        split_tensor_88 = torch.split(view_610,128,dim = -1) ;  view_610 = None
        getitem_464 = split_tensor_88[0]
        getitem_465 = split_tensor_88[1];  split_tensor_88 = None
        add_196 = torch.add(getitem_464,1) ;  getitem_464 = None
        mul_148 = torch.mul(getitem_461,add_196) ;  getitem_461 = add_196 = None
        add_197 = torch.add(mul_148,getitem_465) ;  mul_148 = getitem_465 = None
        t_199 = arg313_1.t() ;  arg313_1 = None
        view_611 = add_197.view(58880, 128) ;  add_197 = None
        mm_167 = torch.mm(view_611,t_199) ;  view_611 = t_199 = None
        view_612 = mm_167.view(5, 11776, 512) ;  mm_167 = None
        split_tensor_89 = torch.split(view_612,256,dim = -1) ;  view_612 = None
        getitem_466 = split_tensor_89[0]
        getitem_467 = split_tensor_89[1];  split_tensor_89 = None
        silu_25 = torch.nn.functional.silu(getitem_466) ;  getitem_466 = None
        mul_149 = torch.mul(silu_25,getitem_467) ;  silu_25 = getitem_467 = None
        t_200 = arg315_1.t() ;  arg315_1 = None
        clone_107 = torch.clone(view_546,memory_format = torch.contiguous_format) ;  view_546 = None
        _unsafe_view_105 = clone_107.view(58880, 128) ;  clone_107 = None
        mm_168 = torch.mm(_unsafe_view_105,t_200) ;  _unsafe_view_105 = t_200 = None
        view_613 = mm_168.view(5, 11776, 128) ;  mm_168 = None
        add_198 = torch.add(view_613,arg316_1) ;  view_613 = arg316_1 = None
        sigmoid_43 = torch.sigmoid(add_198) ;  add_198 = None
        t_201 = arg314_1.t() ;  arg314_1 = None
        view_614 = mul_149.view(58880, 256) ;  mul_149 = None
        mm_169 = torch.mm(view_614,t_201) ;  view_614 = t_201 = None
        view_615 = mm_169.view(5, 11776, 128) ;  mm_169 = None
        mul_150 = torch.mul(sigmoid_43,view_615) ;  sigmoid_43 = view_615 = None
        add_199 = torch.add(masked_fill_42,mul_150) ;  masked_fill_42 = mul_150 = None
        add_200 = torch.add(add_199,mul_147) ;  add_199 = mul_147 = None
        native_layer_norm_default_76 = (torch.nn.functional.layer_norm(add_200,[128],arg335_1,arg336_1,1e-05),) ;  add_200 = arg335_1 = arg336_1 = None
        getitem_468 = native_layer_norm_default_76[0]
        t_202 = arg337_1.t() ;  arg337_1 = None
        view_616 = getitem_468.view(58880, 128) ;  getitem_468 = None
        mm_170 = torch.mm(view_616,t_202) ;  view_616 = t_202 = None
        view_617 = mm_170.view(5, 11776, 3) ;  mm_170 = None
        view_618 = view_617.view(1, 5, 11776, 3) ;  view_617 = None
        mul_151 = torch.mul(view_31,16.0) 
        pow_3 = torch.pow(view_31,2) 
        add_201 = torch.add(pow_3,256.0) ;  pow_3 = None
        pow_4 = torch.pow(add_201,-0.5) ;  add_201 = None
        mul_152 = torch.mul(mul_151,pow_4) ;  mul_151 = pow_4 = None
        mul_153 = torch.mul(view_618,mul_152) ;  view_618 = mul_152 = None
        pow_5 = torch.pow(view_31,2) ;  view_31 = None
        add_202 = torch.add(pow_5,256.0) ;  pow_5 = None
        reciprocal = torch.reciprocal(add_202) ;  add_202 = None
        mul_154 = torch.mul(reciprocal,256.0) ;  reciprocal = None
        mul_155 = torch.mul(arg354_1,mul_154) ;  arg354_1 = mul_154 = None
        add_203 = torch.add(mul_155,mul_153) ;  mul_155 = mul_153 = None
        view_619 = add_203.view(5, 11776, 3) ;  add_203 = None
        return view_619
        
    # To see more debug info, please use `graph_module.print_readable()`