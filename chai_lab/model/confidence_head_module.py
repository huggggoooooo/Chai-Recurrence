import torch
import torch.nn as nn

class confidence_head_module(nn.Module):
    def  __init__(self, model):
        super(confidence_head_module, self).__init__()
        self.device = torch.device('cuda')
        self.arg0_1 = model.single_to_pair_proj.weight
        self.arg1_1 = model.atom_distance_bins_projection.weight
        self.arg2_1 = getattr(model.blocks, "0").transition_pair.layer_norm.weight
        self.arg3_1 = getattr(model.blocks, "0").transition_pair.layer_norm.bias
        self.arg4_1 = getattr(model.blocks, "0").transition_pair.linear_no_bias_ab.weight
        self.arg5_1 = getattr(model.blocks, "0").transition_pair.linear_out.weight
        self.arg6_1 = getattr(model.blocks, "0").triangle_multiplication.layernorm_z_in.weight
        self.arg7_1 = getattr(model.blocks, "0").triangle_multiplication.layernorm_z_in.bias
        self.arg8_1 = getattr(model.blocks, "0").triangle_multiplication.linear_z_out.weight
        self.arg9_1 = getattr(model.blocks, "0").triangle_multiplication.merged_linear_p.weight
        self.arg10_1 = getattr(model.blocks, "0").triangle_multiplication.merged_linear_g.weight
        self.arg11_1 = getattr(model.blocks, "0").triangle_attention.pair_layer_norm.weight
        self.arg12_1 = getattr(model.blocks, "0").triangle_attention.pair_layer_norm.bias
        self.arg13_1 = getattr(model.blocks, "0").triangle_attention.pair2qkvgb.weight
        self.arg14_1 = getattr(model.blocks, "0").triangle_attention.linear_out.weight
        self.arg15_1 = getattr(model.blocks, "0").transition_single.layer_norm.weight
        self.arg16_1 = getattr(model.blocks, "0").transition_single.layer_norm.bias
        self.arg17_1 = getattr(model.blocks, "0").transition_single.linear_no_bias_ab.weight
        self.arg18_1 = getattr(model.blocks, "0").transition_single.linear_out.weight
        self.arg19_1 = getattr(model.blocks, "0").attention_pair_bias.single_layer_norm.weight
        self.arg20_1 = getattr(model.blocks, "0").attention_pair_bias.single_layer_norm.bias
        self.arg21_1 = getattr(model.blocks, "0").attention_pair_bias.pair_layer_norm.weight
        self.arg22_1 = getattr(model.blocks, "0").attention_pair_bias.pair_layer_norm.bias
        self.arg23_1 = getattr(model.blocks, "0").attention_pair_bias.pair_linear.weight
        self.arg24_1 = getattr(model.blocks, "0").attention_pair_bias.attention.query_bias
        self.arg25_1 = getattr(model.blocks, "0").attention_pair_bias.attention.input2qkvg.weight
        self.arg26_1 = getattr(model.blocks, "0").attention_pair_bias.attention.output_proj.weight
        self.arg27_1 = getattr(model.blocks, "1").transition_pair.layer_norm.weight
        self.arg28_1 = getattr(model.blocks, "1").transition_pair.layer_norm.bias
        self.arg29_1 = getattr(model.blocks, "1").transition_pair.linear_no_bias_ab.weight
        self.arg30_1 = getattr(model.blocks, "1").transition_pair.linear_out.weight
        self.arg31_1 = getattr(model.blocks, "1").triangle_multiplication.layernorm_z_in.weight
        self.arg32_1 = getattr(model.blocks, "1").triangle_multiplication.layernorm_z_in.bias
        self.arg33_1 = getattr(model.blocks, "1").triangle_multiplication.linear_z_out.weight
        self.arg34_1 = getattr(model.blocks, "1").triangle_multiplication.merged_linear_p.weight
        self.arg35_1 = getattr(model.blocks, "1").triangle_multiplication.merged_linear_g.weight
        self.arg36_1 = getattr(model.blocks, "1").triangle_attention.pair_layer_norm.weight
        self.arg37_1 = getattr(model.blocks, "1").triangle_attention.pair_layer_norm.bias
        self.arg38_1 = getattr(model.blocks, "1").triangle_attention.pair2qkvgb.weight
        self.arg39_1 = getattr(model.blocks, "1").triangle_attention.linear_out.weight
        self.arg40_1 = getattr(model.blocks, "1").transition_single.layer_norm.weight
        self.arg41_1 = getattr(model.blocks, "1").transition_single.layer_norm.bias
        self.arg42_1 = getattr(model.blocks, "1").transition_single.linear_no_bias_ab.weight
        self.arg43_1 = getattr(model.blocks, "1").transition_single.linear_out.weight
        self.arg44_1 = getattr(model.blocks, "1").attention_pair_bias.single_layer_norm.weight
        self.arg45_1 = getattr(model.blocks, "1").attention_pair_bias.single_layer_norm.bias
        self.arg46_1 = getattr(model.blocks, "1").attention_pair_bias.pair_layer_norm.weight
        self.arg47_1 = getattr(model.blocks, "1").attention_pair_bias.pair_layer_norm.bias
        self.arg48_1 = getattr(model.blocks, "1").attention_pair_bias.pair_linear.weight
        self.arg49_1 = getattr(model.blocks, "1").attention_pair_bias.attention.query_bias
        self.arg50_1 = getattr(model.blocks, "1").attention_pair_bias.attention.input2qkvg.weight
        self.arg51_1 = getattr(model.blocks, "1").attention_pair_bias.attention.output_proj.weight
        self.arg52_1 = getattr(model.blocks, "2").transition_pair.layer_norm.weight
        self.arg53_1 = getattr(model.blocks, "2").transition_pair.layer_norm.bias
        self.arg54_1 = getattr(model.blocks, "2").transition_pair.linear_no_bias_ab.weight
        self.arg55_1 = getattr(model.blocks, "2").transition_pair.linear_out.weight
        self.arg56_1 = getattr(model.blocks, "2").triangle_multiplication.layernorm_z_in.weight
        self.arg57_1 = getattr(model.blocks, "2").triangle_multiplication.layernorm_z_in.bias
        self.arg58_1 = getattr(model.blocks, "2").triangle_multiplication.linear_z_out.weight
        self.arg59_1 = getattr(model.blocks, "2").triangle_multiplication.merged_linear_p.weight
        self.arg60_1 = getattr(model.blocks, "2").triangle_multiplication.merged_linear_g.weight
        self.arg61_1 = getattr(model.blocks, "2").triangle_attention.pair_layer_norm.weight
        self.arg62_1 = getattr(model.blocks, "2").triangle_attention.pair_layer_norm.bias
        self.arg63_1 = getattr(model.blocks, "2").triangle_attention.pair2qkvgb.weight
        self.arg64_1 = getattr(model.blocks, "2").triangle_attention.linear_out.weight
        self.arg65_1 = getattr(model.blocks, "2").transition_single.layer_norm.weight
        self.arg66_1 = getattr(model.blocks, "2").transition_single.layer_norm.bias
        self.arg67_1 = getattr(model.blocks, "2").transition_single.linear_no_bias_ab.weight
        self.arg68_1 = getattr(model.blocks, "2").transition_single.linear_out.weight
        self.arg69_1 = getattr(model.blocks, "2").attention_pair_bias.single_layer_norm.weight
        self.arg70_1 = getattr(model.blocks, "2").attention_pair_bias.single_layer_norm.bias
        self.arg71_1 = getattr(model.blocks, "2").attention_pair_bias.pair_layer_norm.weight
        self.arg72_1 = getattr(model.blocks, "2").attention_pair_bias.pair_layer_norm.bias
        self.arg73_1 = getattr(model.blocks, "2").attention_pair_bias.pair_linear.weight
        self.arg74_1 = getattr(model.blocks, "2").attention_pair_bias.attention.query_bias
        self.arg75_1 = getattr(model.blocks, "2").attention_pair_bias.attention.input2qkvg.weight
        self.arg76_1 = getattr(model.blocks, "2").attention_pair_bias.attention.output_proj.weight
        self.arg77_1 = getattr(model.blocks, "3").transition_pair.layer_norm.weight
        self.arg78_1 = getattr(model.blocks, "3").transition_pair.layer_norm.bias
        self.arg79_1 = getattr(model.blocks, "3").transition_pair.linear_no_bias_ab.weight
        self.arg80_1 = getattr(model.blocks, "3").transition_pair.linear_out.weight
        self.arg81_1 = getattr(model.blocks, "3").triangle_multiplication.layernorm_z_in.weight
        self.arg82_1 = getattr(model.blocks, "3").triangle_multiplication.layernorm_z_in.bias
        self.arg83_1 = getattr(model.blocks, "3").triangle_multiplication.linear_z_out.weight
        self.arg84_1 = getattr(model.blocks, "3").triangle_multiplication.merged_linear_p.weight
        self.arg85_1 = getattr(model.blocks, "3").triangle_multiplication.merged_linear_g.weight
        self.arg86_1 = getattr(model.blocks, "3").triangle_attention.pair_layer_norm.weight
        self.arg87_1 = getattr(model.blocks, "3").triangle_attention.pair_layer_norm.bias
        self.arg88_1 = getattr(model.blocks, "3").triangle_attention.pair2qkvgb.weight
        self.arg89_1 = getattr(model.blocks, "3").triangle_attention.linear_out.weight
        self.arg90_1 = getattr(model.blocks, "3").transition_single.layer_norm.weight
        self.arg91_1 = getattr(model.blocks, "3").transition_single.layer_norm.bias
        self.arg92_1 = getattr(model.blocks, "3").transition_single.linear_no_bias_ab.weight
        self.arg93_1 = getattr(model.blocks, "3").transition_single.linear_out.weight
        self.arg94_1 = getattr(model.blocks, "3").attention_pair_bias.single_layer_norm.weight
        self.arg95_1 = getattr(model.blocks, "3").attention_pair_bias.single_layer_norm.bias
        self.arg96_1 = getattr(model.blocks, "3").attention_pair_bias.pair_layer_norm.weight
        self.arg97_1 = getattr(model.blocks, "3").attention_pair_bias.pair_layer_norm.bias
        self.arg98_1 = getattr(model.blocks, "3").attention_pair_bias.pair_linear.weight
        self.arg99_1 = getattr(model.blocks, "3").attention_pair_bias.attention.query_bias
        self.arg100_1 = getattr(model.blocks, "3").attention_pair_bias.attention.input2qkvg.weight
        self.arg101_1 = getattr(model.blocks, "3").attention_pair_bias.attention.output_proj.weight
        self.arg102_1 = model.plddt_projection.weight
        self.arg103_1 = model.pae_projection.weight
        self.arg104_1 = model.pde_projection.weight
        arg105_1 = model.atom_distance_v_bins
        self.register_buffer('arg105_1', arg105_1)

    def forward(self, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1):
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

        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.bfloat16);  arg0_1 = None
        t = torch.ops.aten.t.default(_to_copy);  _to_copy = None
        view = torch.ops.aten.view.default(arg106_1, [512, 384]);  arg106_1 = None
        mm = torch.ops.aten.mm.default(view, t);  view = t = None
        view_1 = torch.ops.aten.view.default(mm, [1, 512, 512]);  mm = None
        split_tensor = torch.ops.aten.split.Tensor(view_1, 256, dim = -1);  view_1 = None
        getitem = split_tensor[0]
        getitem_1 = split_tensor[1];  split_tensor = None
        view_2 = torch.ops.aten.view.default(getitem, [1, 512, 1, 256]);  getitem = None
        view_3 = torch.ops.aten.view.default(getitem_1, [1, 1, 512, 256]);  getitem_1 = None
        add = torch.ops.aten.add.Tensor(arg108_1, view_2);  arg108_1 = view_2 = None
        add_1 = torch.ops.aten.add.Tensor(add, view_3);  add = view_3 = None
        arange = torch.ops.aten.arange.default(1, device = self.device, pin_memory = False)
        view_4 = torch.ops.aten.view.default(arange, [1, 1]);  arange = None
        index = torch.ops.aten.index.Tensor(arg111_1, [view_4, arg112_1]);  arg111_1 = view_4 = arg112_1 = None
        _cdist_forward = torch.ops.aten._cdist_forward.default(index, index, 2.0, 2);  index = None
        searchsorted = torch.ops.aten.searchsorted.Tensor(arg105_1, _cdist_forward);  arg105_1 = _cdist_forward = None
        arange_1 = torch.ops.aten.arange.default(16, dtype = torch.int64, layout = torch.strided, device = self.device)
        unsqueeze = torch.ops.aten.unsqueeze.default(searchsorted, -1);  searchsorted = None
        eq = torch.ops.aten.eq.Tensor(unsqueeze, arange_1);  unsqueeze = arange_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(eq, dtype = torch.int64);  eq = None
        _to_copy_2 = torch.ops.aten._to_copy.default(_to_copy_1, dtype = torch.float32);  _to_copy_1 = None
        _to_copy_3 = torch.ops.aten._to_copy.default(_to_copy_2, dtype = torch.bfloat16);  _to_copy_2 = None
        _to_copy_4 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.bfloat16);  arg1_1 = None
        t_1 = torch.ops.aten.t.default(_to_copy_4);  _to_copy_4 = None
        view_5 = torch.ops.aten.view.default(_to_copy_3, [262144, 16]);  _to_copy_3 = None
        mm_1 = torch.ops.aten.mm.default(view_5, t_1);  view_5 = t_1 = None
        view_6 = torch.ops.aten.view.default(mm_1, [1, 512, 512, 256]);  mm_1 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, view_6);  add_1 = view_6 = None
        view_7 = torch.ops.aten.view.default(arg109_1, [1, 512, 1])
        view_8 = torch.ops.aten.view.default(arg109_1, [1, 1, 512])
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(view_7, view_8);  view_7 = view_8 = None
        _to_copy_5 = torch.ops.aten._to_copy.default(add_2, dtype = torch.float32)
        native_layer_norm_default = torch.ops.aten.native_layer_norm.default(_to_copy_5, [256], arg6_1, arg7_1, 1e-05);  _to_copy_5 = arg6_1 = arg7_1 = None
        getitem_2 = native_layer_norm_default[0]
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(arg9_1, [512, 512]);  arg9_1 = None
        getitem_5 = split_with_sizes_default[0]
        getitem_6 = split_with_sizes_default[1];  split_with_sizes_default = None
        split_with_sizes_default_1 = torch.ops.aten.split_with_sizes.default(arg10_1, [512, 512, 256]);  arg10_1 = None
        getitem_7 = split_with_sizes_default_1[0]
        getitem_8 = split_with_sizes_default_1[1]
        getitem_9 = split_with_sizes_default_1[2];  split_with_sizes_default_1 = None
        _to_copy_6 = torch.ops.aten._to_copy.default(getitem_5, dtype = torch.bfloat16);  getitem_5 = None
        _to_copy_7 = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.bfloat16)
        t_2 = torch.ops.aten.t.default(_to_copy_6);  _to_copy_6 = None
        view_9 = torch.ops.aten.view.default(_to_copy_7, [262144, 256]);  _to_copy_7 = None
        mm_2 = torch.ops.aten.mm.default(view_9, t_2);  view_9 = t_2 = None
        view_10 = torch.ops.aten.view.default(mm_2, [1, 512, 512, 512]);  mm_2 = None
        _to_copy_8 = torch.ops.aten._to_copy.default(getitem_7, dtype = torch.bfloat16);  getitem_7 = None
        _to_copy_9 = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.bfloat16)
        t_3 = torch.ops.aten.t.default(_to_copy_8);  _to_copy_8 = None
        view_11 = torch.ops.aten.view.default(_to_copy_9, [262144, 256]);  _to_copy_9 = None
        mm_3 = torch.ops.aten.mm.default(view_11, t_3);  view_11 = t_3 = None
        view_12 = torch.ops.aten.view.default(mm_3, [1, 512, 512, 512]);  mm_3 = None
        sigmoid = torch.ops.aten.sigmoid.default(view_12);  view_12 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_10, sigmoid);  view_10 = sigmoid = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(bitwise_and_1, 3)
        bitwise_not = torch.ops.aten.bitwise_not.default(unsqueeze_1);  unsqueeze_1 = None
        masked_fill = torch.ops.aten.masked_fill.Scalar(mul_2, bitwise_not, 0);  mul_2 = bitwise_not = None
        split_tensor_1 = torch.ops.aten.split.Tensor(masked_fill, 256, dim = -1)
        getitem_12 = split_tensor_1[0]
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(getitem_12, 4);  getitem_12 = None
        permute_4 = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 4, 3, 2]);  unsqueeze_4 = None
        permute_5 = torch.ops.aten.permute.default(permute_4, [3, 1, 4, 0, 2]);  permute_4 = None
        view_15 = torch.ops.aten.view.default(permute_5, [256, 512, 512]);  permute_5 = None
        split_tensor_2 = torch.ops.aten.split.Tensor(masked_fill, 256, dim = -1);  masked_fill = None
        getitem_15 = split_tensor_2[1];  split_tensor_2 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(getitem_15, 4);  getitem_15 = None
        permute_6 = torch.ops.aten.permute.default(unsqueeze_5, [0, 4, 1, 3, 2]);  unsqueeze_5 = None
        permute_7 = torch.ops.aten.permute.default(permute_6, [3, 4, 0, 2, 1]);  permute_6 = None
        view_16 = torch.ops.aten.view.default(permute_7, [256, 512, 512]);  permute_7 = None
        bmm = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17 = torch.ops.aten.view.default(bmm, [256, 512, 1, 1, 512]);  bmm = None
        permute_8 = torch.ops.aten.permute.default(view_17, [3, 1, 4, 0, 2]);  view_17 = None
        view_18 = torch.ops.aten.view.default(permute_8, [1, 512, 512, 256]);  permute_8 = None
        _to_copy_10 = torch.ops.aten._to_copy.default(getitem_6, dtype = torch.bfloat16);  getitem_6 = None
        _to_copy_11 = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.bfloat16)
        t_4 = torch.ops.aten.t.default(_to_copy_10);  _to_copy_10 = None
        view_19 = torch.ops.aten.view.default(_to_copy_11, [262144, 256]);  _to_copy_11 = None
        mm_4 = torch.ops.aten.mm.default(view_19, t_4);  view_19 = t_4 = None
        view_20 = torch.ops.aten.view.default(mm_4, [1, 512, 512, 512]);  mm_4 = None
        _to_copy_12 = torch.ops.aten._to_copy.default(getitem_8, dtype = torch.bfloat16);  getitem_8 = None
        _to_copy_13 = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.bfloat16)
        t_5 = torch.ops.aten.t.default(_to_copy_12);  _to_copy_12 = None
        view_21 = torch.ops.aten.view.default(_to_copy_13, [262144, 256]);  _to_copy_13 = None
        mm_5 = torch.ops.aten.mm.default(view_21, t_5);  view_21 = t_5 = None
        view_22 = torch.ops.aten.view.default(mm_5, [1, 512, 512, 512]);  mm_5 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(view_22);  view_22 = None
        mul_3 = torch.ops.aten.mul.Tensor(view_20, sigmoid_1);  view_20 = sigmoid_1 = None
        view_23 = torch.ops.aten.view.default(mul_3, [262144, 512]);  mul_3 = None
        view_24 = torch.ops.aten.view.default(view_23, [1, 512, 512, 512]);  view_23 = None
        transpose = torch.ops.aten.transpose.int(bitwise_and_1, 1, 2)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(transpose, 3);  transpose = None
        clone = torch.ops.aten.clone.default(unsqueeze_6, memory_format = torch.contiguous_format);  unsqueeze_6 = None
        bitwise_not_1 = torch.ops.aten.bitwise_not.default(clone);  clone = None
        masked_fill_1 = torch.ops.aten.masked_fill.Scalar(view_24, bitwise_not_1, 0);  view_24 = bitwise_not_1 = None
        view_25 = torch.ops.aten.view.default(masked_fill_1, [262144, 512]);  masked_fill_1 = None
        view_29 = torch.ops.aten.view.default(view_25, [1, 512, 512, 512])
        split_tensor_3 = torch.ops.aten.split.Tensor(view_29, 256, dim = -1);  view_29 = None
        getitem_18 = split_tensor_3[0]
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(getitem_18, 4);  getitem_18 = None
        permute_13 = torch.ops.aten.permute.default(unsqueeze_9, [0, 2, 4, 3, 1]);  unsqueeze_9 = None
        permute_14 = torch.ops.aten.permute.default(permute_13, [3, 1, 4, 0, 2]);  permute_13 = None
        view_30 = torch.ops.aten.view.default(permute_14, [256, 512, 512]);  permute_14 = None
        view_31 = torch.ops.aten.view.default(view_25, [1, 512, 512, 512]);  view_25 = None
        split_tensor_4 = torch.ops.aten.split.Tensor(view_31, 256, dim = -1);  view_31 = None
        getitem_21 = split_tensor_4[1];  split_tensor_4 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(getitem_21, 4);  getitem_21 = None
        permute_15 = torch.ops.aten.permute.default(unsqueeze_10, [0, 4, 2, 3, 1]);  unsqueeze_10 = None
        permute_16 = torch.ops.aten.permute.default(permute_15, [3, 4, 0, 2, 1]);  permute_15 = None
        view_32 = torch.ops.aten.view.default(permute_16, [256, 512, 512]);  permute_16 = None
        bmm_1 = torch.ops.aten.bmm.default(view_30, view_32);  view_30 = view_32 = None
        view_33 = torch.ops.aten.view.default(bmm_1, [256, 512, 1, 1, 512]);  bmm_1 = None
        permute_17 = torch.ops.aten.permute.default(view_33, [3, 1, 4, 0, 2]);  view_33 = None
        view_34 = torch.ops.aten.view.default(permute_17, [1, 512, 512, 256]);  permute_17 = None
        _to_copy_14 = torch.ops.aten._to_copy.default(view_18, dtype = torch.float32);  view_18 = None
        native_layer_norm_default_1 = torch.ops.aten.native_layer_norm.default(_to_copy_14, [256], None, None, 1e-05);  _to_copy_14 = None
        getitem_22 = native_layer_norm_default_1[0]
        _to_copy_15 = torch.ops.aten._to_copy.default(view_34, dtype = torch.float32);  view_34 = None
        native_layer_norm_default_2 = torch.ops.aten.native_layer_norm.default(_to_copy_15, [256], None, None, 1e-05);  _to_copy_15 = None
        getitem_25 = native_layer_norm_default_2[0]
        add_3 = torch.ops.aten.add.Tensor(getitem_22, getitem_25);  getitem_22 = getitem_25 = None
        _to_copy_16 = torch.ops.aten._to_copy.default(arg8_1, dtype = torch.bfloat16);  arg8_1 = None
        _to_copy_17 = torch.ops.aten._to_copy.default(add_3, dtype = torch.bfloat16);  add_3 = None
        t_6 = torch.ops.aten.t.default(_to_copy_16);  _to_copy_16 = None
        view_35 = torch.ops.aten.view.default(_to_copy_17, [262144, 256]);  _to_copy_17 = None
        mm_6 = torch.ops.aten.mm.default(view_35, t_6);  view_35 = t_6 = None
        view_36 = torch.ops.aten.view.default(mm_6, [1, 512, 512, 256]);  mm_6 = None
        _to_copy_18 = torch.ops.aten._to_copy.default(getitem_9, dtype = torch.bfloat16);  getitem_9 = None
        _to_copy_19 = torch.ops.aten._to_copy.default(getitem_2, dtype = torch.bfloat16);  getitem_2 = None
        t_7 = torch.ops.aten.t.default(_to_copy_18);  _to_copy_18 = None
        view_37 = torch.ops.aten.view.default(_to_copy_19, [262144, 256]);  _to_copy_19 = None
        mm_7 = torch.ops.aten.mm.default(view_37, t_7);  view_37 = t_7 = None
        view_38 = torch.ops.aten.view.default(mm_7, [1, 512, 512, 256]);  mm_7 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(view_38);  view_38 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_36, sigmoid_2);  view_36 = sigmoid_2 = None
        add_4 = torch.ops.aten.add.Tensor(add_2, mul_4);  mul_4 = None
        _to_copy_20 = torch.ops.aten._to_copy.default(add_2, dtype = torch.float32)
        native_layer_norm_default_3 = torch.ops.aten.native_layer_norm.default(_to_copy_20, [256], arg11_1, arg12_1, 1e-05);  _to_copy_20 = arg11_1 = arg12_1 = None
        getitem_28 = native_layer_norm_default_3[0]
        _to_copy_21 = torch.ops.aten._to_copy.default(arg13_1, dtype = torch.bfloat16);  arg13_1 = None
        _to_copy_22 = torch.ops.aten._to_copy.default(getitem_28, dtype = torch.bfloat16);  getitem_28 = None
        t_8 = torch.ops.aten.t.default(_to_copy_21);  _to_copy_21 = None
        view_39 = torch.ops.aten.view.default(_to_copy_22, [262144, 256]);  _to_copy_22 = None
        mm_8 = torch.ops.aten.mm.default(view_39, t_8);  view_39 = t_8 = None
        view_40 = torch.ops.aten.view.default(mm_8, [1, 512, 512, 2056]);  mm_8 = None
        split_with_sizes_default_2 = torch.ops.aten.split_with_sizes.default(view_40, [2048, 8], dim = -1);  view_40 = None
        getitem_31 = split_with_sizes_default_2[0]
        getitem_32 = split_with_sizes_default_2[1];  split_with_sizes_default_2 = None
        view_41 = torch.ops.aten.view.default(getitem_32, [1, 512, 512, 2, 4]);  getitem_32 = None
        permute_18 = torch.ops.aten.permute.default(view_41, [0, 3, 4, 1, 2]);  view_41 = None
        view_42 = torch.ops.aten.view.default(permute_18, [1, 2, 4, 1, 512, 512]);  permute_18 = None
        view_43 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 1, 512, 512])
        bitwise_not_2 = torch.ops.aten.bitwise_not.default(view_43);  view_43 = None
        masked_fill_2 = torch.ops.aten.masked_fill.Scalar(view_42, bitwise_not_2, -10000);  view_42 = bitwise_not_2 = None
        view_44 = torch.ops.aten.view.default(masked_fill_2, [1, 2, 4, 512, 512]);  masked_fill_2 = None
        view_45 = torch.ops.aten.view.default(view_44, [8, 1, 512, 512]);  view_44 = None
        split_tensor_5 = torch.ops.aten.split.Tensor(getitem_31, 1024, dim = -1);  getitem_31 = None
        getitem_33 = split_tensor_5[0]
        getitem_34 = split_tensor_5[1];  split_tensor_5 = None
        permute_19 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        stack = torch.ops.aten.stack.default([getitem_33, permute_19]);  getitem_33 = permute_19 = None
        view_46 = torch.ops.aten.view.default(stack, [2, 1, 512, 512, 4, 4, 64]);  stack = None
        permute_20 = torch.ops.aten.permute.default(view_46, [4, 1, 0, 5, 2, 3, 6]);  view_46 = None
        clone_1 = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        _unsafe_view = torch.ops.aten._unsafe_view.default(clone_1, [4, 8, 512, 512, 64]);  clone_1 = None
        unbind_int = torch.ops.aten.unbind.int(_unsafe_view);  _unsafe_view = None
        getitem_35 = unbind_int[0]
        getitem_36 = unbind_int[1]
        getitem_37 = unbind_int[2]
        getitem_38 = unbind_int[3];  unbind_int = None
        split_tensor_6 = torch.ops.aten.split.Tensor(getitem_35, 4);  getitem_35 = None
        getitem_39 = split_tensor_6[0]
        getitem_40 = split_tensor_6[1];  split_tensor_6 = None
        split_tensor_7 = torch.ops.aten.split.Tensor(getitem_36, 4);  getitem_36 = None
        getitem_41 = split_tensor_7[0]
        getitem_42 = split_tensor_7[1];  split_tensor_7 = None
        split_tensor_8 = torch.ops.aten.split.Tensor(getitem_37, 4);  getitem_37 = None
        getitem_43 = split_tensor_8[0]
        getitem_44 = split_tensor_8[1];  split_tensor_8 = None
        split_tensor_9 = torch.ops.aten.split.Tensor(view_45, 4);  view_45 = None
        getitem_45 = split_tensor_9[0]
        getitem_46 = split_tensor_9[1];  split_tensor_9 = None
        expand = torch.ops.aten.expand.default(getitem_45, [4, 512, 512, 512]);  getitem_45 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_39, getitem_41, getitem_43, expand, False);  getitem_39 = getitem_41 = getitem_43 = expand = None
        getitem_47 = _scaled_dot_product_efficient_attention_default[0]
        expand_1 = torch.ops.aten.expand.default(getitem_46, [4, 512, 512, 512]);  getitem_46 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_40, getitem_42, getitem_44, expand_1, False);  getitem_40 = getitem_42 = getitem_44 = expand_1 = None
        getitem_51 = _scaled_dot_product_efficient_attention_default_1[0]
        cat = torch.ops.aten.cat.default([getitem_47, getitem_51]);  getitem_47 = getitem_51 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(getitem_38);  getitem_38 = None
        mul_5 = torch.ops.aten.mul.Tensor(cat, sigmoid_3);  cat = sigmoid_3 = None
        view_47 = torch.ops.aten.view.default(mul_5, [1, 2, 4, 512, 512, 64]);  mul_5 = None
        permute_21 = torch.ops.aten.permute.default(view_47, [0, 3, 4, 1, 2, 5]);  view_47 = None
        clone_2 = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_2, [1, 512, 512, 512]);  clone_2 = None
        _to_copy_23 = torch.ops.aten._to_copy.default(arg14_1, dtype = torch.bfloat16);  arg14_1 = None
        t_9 = torch.ops.aten.t.default(_to_copy_23);  _to_copy_23 = None
        view_48 = torch.ops.aten.view.default(_unsafe_view_1, [262144, 512]);  _unsafe_view_1 = None
        mm_9 = torch.ops.aten.mm.default(view_48, t_9);  view_48 = t_9 = None
        view_49 = torch.ops.aten.view.default(mm_9, [1, 512, 512, 512]);  mm_9 = None
        view_50 = torch.ops.aten.view.default(view_49, [1, 512, 512, 2, 4, 64]);  view_49 = None
        permute_22 = torch.ops.aten.permute.default(view_50, [3, 0, 1, 2, 4, 5]);  view_50 = None
        view_51 = torch.ops.aten.view.default(permute_22, [2, 1, 512, 512, 256]);  permute_22 = None
        unbind_int_1 = torch.ops.aten.unbind.int(view_51);  view_51 = None
        getitem_55 = unbind_int_1[0]
        getitem_56 = unbind_int_1[1];  unbind_int_1 = None
        permute_23 = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        permute_24 = torch.ops.aten.permute.default(permute_23, [0, 2, 1, 3]);  permute_23 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_55, permute_24);  getitem_55 = permute_24 = None
        add_6 = torch.ops.aten.add.Tensor(add_4, add_5);  add_4 = add_5 = None
        split_tensor_10 = torch.ops.aten.split.Tensor(add_2, 512, dim = -2)
        getitem_57 = split_tensor_10[0];  split_tensor_10 = None
        _to_copy_24 = torch.ops.aten._to_copy.default(getitem_57, dtype = torch.float32);  getitem_57 = None
        native_layer_norm_default_4 = torch.ops.aten.native_layer_norm.default(_to_copy_24, [256], arg2_1, arg3_1, 1e-05);  _to_copy_24 = arg2_1 = arg3_1 = None
        getitem_58 = native_layer_norm_default_4[0]
        _to_copy_25 = torch.ops.aten._to_copy.default(arg4_1, dtype = torch.bfloat16);  arg4_1 = None
        _to_copy_26 = torch.ops.aten._to_copy.default(getitem_58, dtype = torch.bfloat16);  getitem_58 = None
        t_10 = torch.ops.aten.t.default(_to_copy_25);  _to_copy_25 = None
        view_52 = torch.ops.aten.view.default(_to_copy_26, [262144, 256]);  _to_copy_26 = None
        mm_10 = torch.ops.aten.mm.default(view_52, t_10);  view_52 = t_10 = None
        view_53 = torch.ops.aten.view.default(mm_10, [1, 512, 512, 1024]);  mm_10 = None
        split_tensor_11 = torch.ops.aten.split.Tensor(view_53, 512, dim = -1);  view_53 = None
        getitem_61 = split_tensor_11[0]
        getitem_62 = split_tensor_11[1];  split_tensor_11 = None
        silu = torch.ops.aten.silu.default(getitem_61);  getitem_61 = None
        mul_6 = torch.ops.aten.mul.Tensor(silu, getitem_62);  silu = getitem_62 = None
        _to_copy_27 = torch.ops.aten._to_copy.default(arg5_1, dtype = torch.bfloat16);  arg5_1 = None
        t_11 = torch.ops.aten.t.default(_to_copy_27);  _to_copy_27 = None
        view_55 = torch.ops.aten.view.default(mul_6, [262144, 512]);  mul_6 = None
        mm_11 = torch.ops.aten.mm.default(view_55, t_11);  view_55 = t_11 = None
        view_56 = torch.ops.aten.view.default(mm_11, [1, 512, 512, 256]);  mm_11 = None
        add_7 = torch.ops.aten.add.Tensor(add_6, view_56);  add_6 = view_56 = None
        _to_copy_28 = torch.ops.aten._to_copy.default(arg107_1, dtype = torch.float32)
        native_layer_norm_default_5 = torch.ops.aten.native_layer_norm.default(_to_copy_28, [384], arg19_1, arg20_1, 1e-05);  _to_copy_28 = arg19_1 = arg20_1 = None
        getitem_63 = native_layer_norm_default_5[0]
        _to_copy_29 = torch.ops.aten._to_copy.default(add_2, dtype = torch.float32);  add_2 = None
        native_layer_norm_default_6 = torch.ops.aten.native_layer_norm.default(_to_copy_29, [256], arg21_1, arg22_1, 1e-05);  _to_copy_29 = arg21_1 = arg22_1 = None
        getitem_66 = native_layer_norm_default_6[0]
        _to_copy_30 = torch.ops.aten._to_copy.default(arg23_1, dtype = torch.bfloat16);  arg23_1 = None
        _to_copy_31 = torch.ops.aten._to_copy.default(getitem_66, dtype = torch.bfloat16);  getitem_66 = None
        t_12 = torch.ops.aten.t.default(_to_copy_30);  _to_copy_30 = None
        view_57 = torch.ops.aten.view.default(_to_copy_31, [262144, 256]);  _to_copy_31 = None
        mm_12 = torch.ops.aten.mm.default(view_57, t_12);  view_57 = t_12 = None
        view_58 = torch.ops.aten.view.default(mm_12, [1, 512, 512, 16]);  mm_12 = None
        permute_25 = torch.ops.aten.permute.default(view_58, [0, 3, 1, 2]);  view_58 = None
        view_59 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 512, 512]);  bitwise_and_1 = None
        bitwise_not_3 = torch.ops.aten.bitwise_not.default(view_59);  view_59 = None
        masked_fill_3 = torch.ops.aten.masked_fill.Scalar(permute_25, bitwise_not_3, -10000);  permute_25 = bitwise_not_3 = None
        _to_copy_32 = torch.ops.aten._to_copy.default(getitem_63, dtype = torch.bfloat16);  getitem_63 = None
        _to_copy_33 = torch.ops.aten._to_copy.default(arg25_1, dtype = torch.bfloat16);  arg25_1 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(_to_copy_32, 3);  _to_copy_32 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 5);  unsqueeze_12 = None
        permute_26 = torch.ops.aten.permute.default(unsqueeze_13, [3, 0, 4, 1, 5, 2]);  unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(_to_copy_33, 4);  _to_copy_33 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 5);  unsqueeze_14 = None
        permute_27 = torch.ops.aten.permute.default(unsqueeze_15, [1, 4, 2, 5, 3, 0]);  unsqueeze_15 = None
        permute_28 = torch.ops.aten.permute.default(permute_26, [3, 5, 0, 1, 2, 4]);  permute_26 = None
        view_60 = torch.ops.aten.view.default(permute_28, [1, 512, 384]);  permute_28 = None
        permute_29 = torch.ops.aten.permute.default(permute_27, [5, 0, 1, 2, 4, 3]);  permute_27 = None
        view_61 = torch.ops.aten.view.default(permute_29, [1, 384, 1536]);  permute_29 = None
        bmm_2 = torch.ops.aten.bmm.default(view_60, view_61);  view_60 = view_61 = None
        view_62 = torch.ops.aten.view.default(bmm_2, [512, 1, 4, 1, 16, 24]);  bmm_2 = None
        permute_30 = torch.ops.aten.permute.default(view_62, [2, 3, 4, 0, 5, 1]);  view_62 = None
        view_63 = torch.ops.aten.view.default(permute_30, [4, 1, 16, 512, 24]);  permute_30 = None
        unbind_int_2 = torch.ops.aten.unbind.int(view_63);  view_63 = None
        getitem_69 = unbind_int_2[0]
        getitem_70 = unbind_int_2[1]
        getitem_71 = unbind_int_2[2]
        getitem_72 = unbind_int_2[3];  unbind_int_2 = None
        view_64 = torch.ops.aten.view.default(arg24_1, [1, 16, 1, 24]);  arg24_1 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_69, view_64);  getitem_69 = view_64 = None
        _to_copy_34 = torch.ops.aten._to_copy.default(add_8, dtype = torch.bfloat16);  add_8 = None
        expand_2 = torch.ops.aten.expand.default(masked_fill_3, [1, 16, 512, 512]);  masked_fill_3 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(_to_copy_34, getitem_70, getitem_71, expand_2, False);  _to_copy_34 = getitem_70 = getitem_71 = expand_2 = None
        getitem_73 = _scaled_dot_product_efficient_attention_default_2[0]
        add_9 = torch.ops.aten.add.Tensor(getitem_72, 1);  getitem_72 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(add_9);  add_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(getitem_73, sigmoid_4);  getitem_73 = sigmoid_4 = None
        _to_copy_35 = torch.ops.aten._to_copy.default(arg26_1, dtype = torch.bfloat16);  arg26_1 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(mul_7, 4);  mul_7 = None
        permute_31 = torch.ops.aten.permute.default(unsqueeze_16, [0, 2, 4, 3, 1]);  unsqueeze_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(_to_copy_35, 3);  _to_copy_35 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
        permute_32 = torch.ops.aten.permute.default(unsqueeze_18, [3, 4, 2, 1, 0]);  unsqueeze_18 = None
        permute_33 = torch.ops.aten.permute.default(permute_31, [1, 3, 4, 0, 2]);  permute_31 = None
        clone_3 = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_3, [1, 512, 384]);  clone_3 = None
        permute_34 = torch.ops.aten.permute.default(permute_32, [3, 4, 0, 2, 1]);  permute_32 = None
        clone_4 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_4, [1, 384, 384]);  clone_4 = None
        bmm_3 = torch.ops.aten.bmm.default(_unsafe_view_2, _unsafe_view_3);  _unsafe_view_2 = _unsafe_view_3 = None
        view_65 = torch.ops.aten.view.default(bmm_3, [512, 1, 1, 1, 384]);  bmm_3 = None
        permute_35 = torch.ops.aten.permute.default(view_65, [3, 0, 4, 1, 2]);  view_65 = None
        view_66 = torch.ops.aten.view.default(permute_35, [1, 512, 384]);  permute_35 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(arg109_1, -1)
        mul_8 = torch.ops.aten.mul.Tensor(view_66, unsqueeze_19);  view_66 = unsqueeze_19 = None
        add_10 = torch.ops.aten.add.Tensor(arg107_1, mul_8);  mul_8 = None
        split_tensor_12 = torch.ops.aten.split.Tensor(arg107_1, 512, dim = -2);  arg107_1 = None
        getitem_77 = split_tensor_12[0];  split_tensor_12 = None
        _to_copy_36 = torch.ops.aten._to_copy.default(getitem_77, dtype = torch.float32);  getitem_77 = None
        native_layer_norm_default_7 = torch.ops.aten.native_layer_norm.default(_to_copy_36, [384], arg15_1, arg16_1, 1e-05);  _to_copy_36 = arg15_1 = arg16_1 = None
        getitem_78 = native_layer_norm_default_7[0]
        _to_copy_37 = torch.ops.aten._to_copy.default(arg17_1, dtype = torch.bfloat16);  arg17_1 = None
        _to_copy_38 = torch.ops.aten._to_copy.default(getitem_78, dtype = torch.bfloat16);  getitem_78 = None
        t_13 = torch.ops.aten.t.default(_to_copy_37);  _to_copy_37 = None
        view_67 = torch.ops.aten.view.default(_to_copy_38, [512, 384]);  _to_copy_38 = None
        mm_13 = torch.ops.aten.mm.default(view_67, t_13);  view_67 = t_13 = None
        view_68 = torch.ops.aten.view.default(mm_13, [1, 512, 1536]);  mm_13 = None
        split_tensor_13 = torch.ops.aten.split.Tensor(view_68, 768, dim = -1);  view_68 = None
        getitem_81 = split_tensor_13[0]
        getitem_82 = split_tensor_13[1];  split_tensor_13 = None
        silu_1 = torch.ops.aten.silu.default(getitem_81);  getitem_81 = None
        mul_9 = torch.ops.aten.mul.Tensor(silu_1, getitem_82);  silu_1 = getitem_82 = None
        _to_copy_39 = torch.ops.aten._to_copy.default(arg18_1, dtype = torch.bfloat16);  arg18_1 = None
        t_14 = torch.ops.aten.t.default(_to_copy_39);  _to_copy_39 = None
        view_70 = torch.ops.aten.view.default(mul_9, [512, 768]);  mul_9 = None
        mm_14 = torch.ops.aten.mm.default(view_70, t_14);  view_70 = t_14 = None
        view_71 = torch.ops.aten.view.default(mm_14, [1, 512, 384]);  mm_14 = None
        add_11 = torch.ops.aten.add.Tensor(add_10, view_71);  add_10 = view_71 = None
        view_72 = torch.ops.aten.view.default(arg109_1, [1, 512, 1])
        view_73 = torch.ops.aten.view.default(arg109_1, [1, 1, 512])
        bitwise_and_2 = torch.ops.aten.bitwise_and.Tensor(view_72, view_73);  view_72 = view_73 = None
        _to_copy_40 = torch.ops.aten._to_copy.default(add_7, dtype = torch.float32)
        native_layer_norm_default_8 = torch.ops.aten.native_layer_norm.default(_to_copy_40, [256], arg31_1, arg32_1, 1e-05);  _to_copy_40 = arg31_1 = arg32_1 = None
        getitem_83 = native_layer_norm_default_8[0]
        split_with_sizes_default_3 = torch.ops.aten.split_with_sizes.default(arg34_1, [512, 512]);  arg34_1 = None
        getitem_86 = split_with_sizes_default_3[0]
        getitem_87 = split_with_sizes_default_3[1];  split_with_sizes_default_3 = None
        split_with_sizes_default_4 = torch.ops.aten.split_with_sizes.default(arg35_1, [512, 512, 256]);  arg35_1 = None
        getitem_88 = split_with_sizes_default_4[0]
        getitem_89 = split_with_sizes_default_4[1]
        getitem_90 = split_with_sizes_default_4[2];  split_with_sizes_default_4 = None
        _to_copy_41 = torch.ops.aten._to_copy.default(getitem_86, dtype = torch.bfloat16);  getitem_86 = None
        _to_copy_42 = torch.ops.aten._to_copy.default(getitem_83, dtype = torch.bfloat16)
        t_15 = torch.ops.aten.t.default(_to_copy_41);  _to_copy_41 = None
        view_74 = torch.ops.aten.view.default(_to_copy_42, [262144, 256]);  _to_copy_42 = None
        mm_15 = torch.ops.aten.mm.default(view_74, t_15);  view_74 = t_15 = None
        view_75 = torch.ops.aten.view.default(mm_15, [1, 512, 512, 512]);  mm_15 = None
        _to_copy_43 = torch.ops.aten._to_copy.default(getitem_88, dtype = torch.bfloat16);  getitem_88 = None
        _to_copy_44 = torch.ops.aten._to_copy.default(getitem_83, dtype = torch.bfloat16)
        t_16 = torch.ops.aten.t.default(_to_copy_43);  _to_copy_43 = None
        view_76 = torch.ops.aten.view.default(_to_copy_44, [262144, 256]);  _to_copy_44 = None
        mm_16 = torch.ops.aten.mm.default(view_76, t_16);  view_76 = t_16 = None
        view_77 = torch.ops.aten.view.default(mm_16, [1, 512, 512, 512]);  mm_16 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(view_77);  view_77 = None
        mul_10 = torch.ops.aten.mul.Tensor(view_75, sigmoid_5);  view_75 = sigmoid_5 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(bitwise_and_2, 3)
        bitwise_not_4 = torch.ops.aten.bitwise_not.default(unsqueeze_20);  unsqueeze_20 = None
        masked_fill_4 = torch.ops.aten.masked_fill.Scalar(mul_10, bitwise_not_4, 0);  mul_10 = bitwise_not_4 = None
        split_tensor_14 = torch.ops.aten.split.Tensor(masked_fill_4, 256, dim = -1)
        getitem_93 = split_tensor_14[0]
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(getitem_93, 4);  getitem_93 = None
        permute_40 = torch.ops.aten.permute.default(unsqueeze_23, [0, 1, 4, 3, 2]);  unsqueeze_23 = None
        permute_41 = torch.ops.aten.permute.default(permute_40, [3, 1, 4, 0, 2]);  permute_40 = None
        view_80 = torch.ops.aten.view.default(permute_41, [256, 512, 512]);  permute_41 = None
        split_tensor_15 = torch.ops.aten.split.Tensor(masked_fill_4, 256, dim = -1);  masked_fill_4 = None
        getitem_96 = split_tensor_15[1];  split_tensor_15 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(getitem_96, 4);  getitem_96 = None
        permute_42 = torch.ops.aten.permute.default(unsqueeze_24, [0, 4, 1, 3, 2]);  unsqueeze_24 = None
        permute_43 = torch.ops.aten.permute.default(permute_42, [3, 4, 0, 2, 1]);  permute_42 = None
        view_81 = torch.ops.aten.view.default(permute_43, [256, 512, 512]);  permute_43 = None
        bmm_4 = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
        view_82 = torch.ops.aten.view.default(bmm_4, [256, 512, 1, 1, 512]);  bmm_4 = None
        permute_44 = torch.ops.aten.permute.default(view_82, [3, 1, 4, 0, 2]);  view_82 = None
        view_83 = torch.ops.aten.view.default(permute_44, [1, 512, 512, 256]);  permute_44 = None
        _to_copy_45 = torch.ops.aten._to_copy.default(getitem_87, dtype = torch.bfloat16);  getitem_87 = None
        _to_copy_46 = torch.ops.aten._to_copy.default(getitem_83, dtype = torch.bfloat16)
        t_17 = torch.ops.aten.t.default(_to_copy_45);  _to_copy_45 = None
        view_84 = torch.ops.aten.view.default(_to_copy_46, [262144, 256]);  _to_copy_46 = None
        mm_17 = torch.ops.aten.mm.default(view_84, t_17);  view_84 = t_17 = None
        view_85 = torch.ops.aten.view.default(mm_17, [1, 512, 512, 512]);  mm_17 = None
        _to_copy_47 = torch.ops.aten._to_copy.default(getitem_89, dtype = torch.bfloat16);  getitem_89 = None
        _to_copy_48 = torch.ops.aten._to_copy.default(getitem_83, dtype = torch.bfloat16)
        t_18 = torch.ops.aten.t.default(_to_copy_47);  _to_copy_47 = None
        view_86 = torch.ops.aten.view.default(_to_copy_48, [262144, 256]);  _to_copy_48 = None
        mm_18 = torch.ops.aten.mm.default(view_86, t_18);  view_86 = t_18 = None
        view_87 = torch.ops.aten.view.default(mm_18, [1, 512, 512, 512]);  mm_18 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(view_87);  view_87 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_85, sigmoid_6);  view_85 = sigmoid_6 = None
        view_88 = torch.ops.aten.view.default(mul_11, [262144, 512]);  mul_11 = None
        view_89 = torch.ops.aten.view.default(view_88, [1, 512, 512, 512]);  view_88 = None
        transpose_1 = torch.ops.aten.transpose.int(bitwise_and_2, 1, 2)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(transpose_1, 3);  transpose_1 = None
        clone_5 = torch.ops.aten.clone.default(unsqueeze_25, memory_format = torch.contiguous_format);  unsqueeze_25 = None
        bitwise_not_5 = torch.ops.aten.bitwise_not.default(clone_5);  clone_5 = None
        masked_fill_5 = torch.ops.aten.masked_fill.Scalar(view_89, bitwise_not_5, 0);  view_89 = bitwise_not_5 = None
        view_90 = torch.ops.aten.view.default(masked_fill_5, [262144, 512]);  masked_fill_5 = None
        view_94 = torch.ops.aten.view.default(view_90, [1, 512, 512, 512])
        split_tensor_16 = torch.ops.aten.split.Tensor(view_94, 256, dim = -1);  view_94 = None
        getitem_99 = split_tensor_16[0]
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(getitem_99, 4);  getitem_99 = None
        permute_49 = torch.ops.aten.permute.default(unsqueeze_28, [0, 2, 4, 3, 1]);  unsqueeze_28 = None
        permute_50 = torch.ops.aten.permute.default(permute_49, [3, 1, 4, 0, 2]);  permute_49 = None
        view_95 = torch.ops.aten.view.default(permute_50, [256, 512, 512]);  permute_50 = None
        view_96 = torch.ops.aten.view.default(view_90, [1, 512, 512, 512]);  view_90 = None
        split_tensor_17 = torch.ops.aten.split.Tensor(view_96, 256, dim = -1);  view_96 = None
        getitem_102 = split_tensor_17[1];  split_tensor_17 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(getitem_102, 4);  getitem_102 = None
        permute_51 = torch.ops.aten.permute.default(unsqueeze_29, [0, 4, 2, 3, 1]);  unsqueeze_29 = None
        permute_52 = torch.ops.aten.permute.default(permute_51, [3, 4, 0, 2, 1]);  permute_51 = None
        view_97 = torch.ops.aten.view.default(permute_52, [256, 512, 512]);  permute_52 = None
        bmm_5 = torch.ops.aten.bmm.default(view_95, view_97);  view_95 = view_97 = None
        view_98 = torch.ops.aten.view.default(bmm_5, [256, 512, 1, 1, 512]);  bmm_5 = None
        permute_53 = torch.ops.aten.permute.default(view_98, [3, 1, 4, 0, 2]);  view_98 = None
        view_99 = torch.ops.aten.view.default(permute_53, [1, 512, 512, 256]);  permute_53 = None
        _to_copy_49 = torch.ops.aten._to_copy.default(view_83, dtype = torch.float32);  view_83 = None
        native_layer_norm_default_9 = torch.ops.aten.native_layer_norm.default(_to_copy_49, [256], None, None, 1e-05);  _to_copy_49 = None
        getitem_103 = native_layer_norm_default_9[0]
        _to_copy_50 = torch.ops.aten._to_copy.default(view_99, dtype = torch.float32);  view_99 = None
        native_layer_norm_default_10 = torch.ops.aten.native_layer_norm.default(_to_copy_50, [256], None, None, 1e-05);  _to_copy_50 = None
        getitem_106 = native_layer_norm_default_10[0]
        add_12 = torch.ops.aten.add.Tensor(getitem_103, getitem_106);  getitem_103 = getitem_106 = None
        _to_copy_51 = torch.ops.aten._to_copy.default(arg33_1, dtype = torch.bfloat16);  arg33_1 = None
        _to_copy_52 = torch.ops.aten._to_copy.default(add_12, dtype = torch.bfloat16);  add_12 = None
        t_19 = torch.ops.aten.t.default(_to_copy_51);  _to_copy_51 = None
        view_100 = torch.ops.aten.view.default(_to_copy_52, [262144, 256]);  _to_copy_52 = None
        mm_19 = torch.ops.aten.mm.default(view_100, t_19);  view_100 = t_19 = None
        view_101 = torch.ops.aten.view.default(mm_19, [1, 512, 512, 256]);  mm_19 = None
        _to_copy_53 = torch.ops.aten._to_copy.default(getitem_90, dtype = torch.bfloat16);  getitem_90 = None
        _to_copy_54 = torch.ops.aten._to_copy.default(getitem_83, dtype = torch.bfloat16);  getitem_83 = None
        t_20 = torch.ops.aten.t.default(_to_copy_53);  _to_copy_53 = None
        view_102 = torch.ops.aten.view.default(_to_copy_54, [262144, 256]);  _to_copy_54 = None
        mm_20 = torch.ops.aten.mm.default(view_102, t_20);  view_102 = t_20 = None
        view_103 = torch.ops.aten.view.default(mm_20, [1, 512, 512, 256]);  mm_20 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(view_103);  view_103 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_101, sigmoid_7);  view_101 = sigmoid_7 = None
        add_13 = torch.ops.aten.add.Tensor(add_7, mul_12);  mul_12 = None
        _to_copy_55 = torch.ops.aten._to_copy.default(add_7, dtype = torch.float32)
        native_layer_norm_default_11 = torch.ops.aten.native_layer_norm.default(_to_copy_55, [256], arg36_1, arg37_1, 1e-05);  _to_copy_55 = arg36_1 = arg37_1 = None
        getitem_109 = native_layer_norm_default_11[0]
        _to_copy_56 = torch.ops.aten._to_copy.default(arg38_1, dtype = torch.bfloat16);  arg38_1 = None
        _to_copy_57 = torch.ops.aten._to_copy.default(getitem_109, dtype = torch.bfloat16);  getitem_109 = None
        t_21 = torch.ops.aten.t.default(_to_copy_56);  _to_copy_56 = None
        view_104 = torch.ops.aten.view.default(_to_copy_57, [262144, 256]);  _to_copy_57 = None
        mm_21 = torch.ops.aten.mm.default(view_104, t_21);  view_104 = t_21 = None
        view_105 = torch.ops.aten.view.default(mm_21, [1, 512, 512, 2056]);  mm_21 = None
        split_with_sizes_default_5 = torch.ops.aten.split_with_sizes.default(view_105, [2048, 8], dim = -1);  view_105 = None
        getitem_112 = split_with_sizes_default_5[0]
        getitem_113 = split_with_sizes_default_5[1];  split_with_sizes_default_5 = None
        view_106 = torch.ops.aten.view.default(getitem_113, [1, 512, 512, 2, 4]);  getitem_113 = None
        permute_54 = torch.ops.aten.permute.default(view_106, [0, 3, 4, 1, 2]);  view_106 = None
        view_107 = torch.ops.aten.view.default(permute_54, [1, 2, 4, 1, 512, 512]);  permute_54 = None
        view_108 = torch.ops.aten.view.default(bitwise_and_2, [1, 1, 1, 1, 512, 512])
        bitwise_not_6 = torch.ops.aten.bitwise_not.default(view_108);  view_108 = None
        masked_fill_6 = torch.ops.aten.masked_fill.Scalar(view_107, bitwise_not_6, -10000);  view_107 = bitwise_not_6 = None
        view_109 = torch.ops.aten.view.default(masked_fill_6, [1, 2, 4, 512, 512]);  masked_fill_6 = None
        view_110 = torch.ops.aten.view.default(view_109, [8, 1, 512, 512]);  view_109 = None
        split_tensor_18 = torch.ops.aten.split.Tensor(getitem_112, 1024, dim = -1);  getitem_112 = None
        getitem_114 = split_tensor_18[0]
        getitem_115 = split_tensor_18[1];  split_tensor_18 = None
        permute_55 = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        stack_1 = torch.ops.aten.stack.default([getitem_114, permute_55]);  getitem_114 = permute_55 = None
        view_111 = torch.ops.aten.view.default(stack_1, [2, 1, 512, 512, 4, 4, 64]);  stack_1 = None
        permute_56 = torch.ops.aten.permute.default(view_111, [4, 1, 0, 5, 2, 3, 6]);  view_111 = None
        clone_6 = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        _unsafe_view_4 = torch.ops.aten._unsafe_view.default(clone_6, [4, 8, 512, 512, 64]);  clone_6 = None
        unbind_int_3 = torch.ops.aten.unbind.int(_unsafe_view_4);  _unsafe_view_4 = None
        getitem_116 = unbind_int_3[0]
        getitem_117 = unbind_int_3[1]
        getitem_118 = unbind_int_3[2]
        getitem_119 = unbind_int_3[3];  unbind_int_3 = None
        split_tensor_19 = torch.ops.aten.split.Tensor(getitem_116, 4);  getitem_116 = None
        getitem_120 = split_tensor_19[0]
        getitem_121 = split_tensor_19[1];  split_tensor_19 = None
        split_tensor_20 = torch.ops.aten.split.Tensor(getitem_117, 4);  getitem_117 = None
        getitem_122 = split_tensor_20[0]
        getitem_123 = split_tensor_20[1];  split_tensor_20 = None
        split_tensor_21 = torch.ops.aten.split.Tensor(getitem_118, 4);  getitem_118 = None
        getitem_124 = split_tensor_21[0]
        getitem_125 = split_tensor_21[1];  split_tensor_21 = None
        split_tensor_22 = torch.ops.aten.split.Tensor(view_110, 4);  view_110 = None
        getitem_126 = split_tensor_22[0]
        getitem_127 = split_tensor_22[1];  split_tensor_22 = None
        expand_3 = torch.ops.aten.expand.default(getitem_126, [4, 512, 512, 512]);  getitem_126 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_120, getitem_122, getitem_124, expand_3, False);  getitem_120 = getitem_122 = getitem_124 = expand_3 = None
        getitem_128 = _scaled_dot_product_efficient_attention_default_3[0]
        expand_4 = torch.ops.aten.expand.default(getitem_127, [4, 512, 512, 512]);  getitem_127 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_121, getitem_123, getitem_125, expand_4, False);  getitem_121 = getitem_123 = getitem_125 = expand_4 = None
        getitem_132 = _scaled_dot_product_efficient_attention_default_4[0]
        cat_1 = torch.ops.aten.cat.default([getitem_128, getitem_132]);  getitem_128 = getitem_132 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(getitem_119);  getitem_119 = None
        mul_13 = torch.ops.aten.mul.Tensor(cat_1, sigmoid_8);  cat_1 = sigmoid_8 = None
        view_112 = torch.ops.aten.view.default(mul_13, [1, 2, 4, 512, 512, 64]);  mul_13 = None
        permute_57 = torch.ops.aten.permute.default(view_112, [0, 3, 4, 1, 2, 5]);  view_112 = None
        clone_7 = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        _unsafe_view_5 = torch.ops.aten._unsafe_view.default(clone_7, [1, 512, 512, 512]);  clone_7 = None
        _to_copy_58 = torch.ops.aten._to_copy.default(arg39_1, dtype = torch.bfloat16);  arg39_1 = None
        t_22 = torch.ops.aten.t.default(_to_copy_58);  _to_copy_58 = None
        view_113 = torch.ops.aten.view.default(_unsafe_view_5, [262144, 512]);  _unsafe_view_5 = None
        mm_22 = torch.ops.aten.mm.default(view_113, t_22);  view_113 = t_22 = None
        view_114 = torch.ops.aten.view.default(mm_22, [1, 512, 512, 512]);  mm_22 = None
        view_115 = torch.ops.aten.view.default(view_114, [1, 512, 512, 2, 4, 64]);  view_114 = None
        permute_58 = torch.ops.aten.permute.default(view_115, [3, 0, 1, 2, 4, 5]);  view_115 = None
        view_116 = torch.ops.aten.view.default(permute_58, [2, 1, 512, 512, 256]);  permute_58 = None
        unbind_int_4 = torch.ops.aten.unbind.int(view_116);  view_116 = None
        getitem_136 = unbind_int_4[0]
        getitem_137 = unbind_int_4[1];  unbind_int_4 = None
        permute_59 = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
        permute_60 = torch.ops.aten.permute.default(permute_59, [0, 2, 1, 3]);  permute_59 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_136, permute_60);  getitem_136 = permute_60 = None
        add_15 = torch.ops.aten.add.Tensor(add_13, add_14);  add_13 = add_14 = None
        split_tensor_23 = torch.ops.aten.split.Tensor(add_7, 512, dim = -2)
        getitem_138 = split_tensor_23[0];  split_tensor_23 = None
        _to_copy_59 = torch.ops.aten._to_copy.default(getitem_138, dtype = torch.float32);  getitem_138 = None
        native_layer_norm_default_12 = torch.ops.aten.native_layer_norm.default(_to_copy_59, [256], arg27_1, arg28_1, 1e-05);  _to_copy_59 = arg27_1 = arg28_1 = None
        getitem_139 = native_layer_norm_default_12[0]
        _to_copy_60 = torch.ops.aten._to_copy.default(arg29_1, dtype = torch.bfloat16);  arg29_1 = None
        _to_copy_61 = torch.ops.aten._to_copy.default(getitem_139, dtype = torch.bfloat16);  getitem_139 = None
        t_23 = torch.ops.aten.t.default(_to_copy_60);  _to_copy_60 = None
        view_117 = torch.ops.aten.view.default(_to_copy_61, [262144, 256]);  _to_copy_61 = None
        mm_23 = torch.ops.aten.mm.default(view_117, t_23);  view_117 = t_23 = None
        view_118 = torch.ops.aten.view.default(mm_23, [1, 512, 512, 1024]);  mm_23 = None
        split_tensor_24 = torch.ops.aten.split.Tensor(view_118, 512, dim = -1);  view_118 = None
        getitem_142 = split_tensor_24[0]
        getitem_143 = split_tensor_24[1];  split_tensor_24 = None
        silu_2 = torch.ops.aten.silu.default(getitem_142);  getitem_142 = None
        mul_14 = torch.ops.aten.mul.Tensor(silu_2, getitem_143);  silu_2 = getitem_143 = None
        _to_copy_62 = torch.ops.aten._to_copy.default(arg30_1, dtype = torch.bfloat16);  arg30_1 = None
        t_24 = torch.ops.aten.t.default(_to_copy_62);  _to_copy_62 = None
        view_120 = torch.ops.aten.view.default(mul_14, [262144, 512]);  mul_14 = None
        mm_24 = torch.ops.aten.mm.default(view_120, t_24);  view_120 = t_24 = None
        view_121 = torch.ops.aten.view.default(mm_24, [1, 512, 512, 256]);  mm_24 = None
        add_16 = torch.ops.aten.add.Tensor(add_15, view_121);  add_15 = view_121 = None
        _to_copy_63 = torch.ops.aten._to_copy.default(add_11, dtype = torch.float32)
        native_layer_norm_default_13 = torch.ops.aten.native_layer_norm.default(_to_copy_63, [384], arg44_1, arg45_1, 1e-05);  _to_copy_63 = arg44_1 = arg45_1 = None
        getitem_144 = native_layer_norm_default_13[0]
        _to_copy_64 = torch.ops.aten._to_copy.default(add_7, dtype = torch.float32);  add_7 = None
        native_layer_norm_default_14 = torch.ops.aten.native_layer_norm.default(_to_copy_64, [256], arg46_1, arg47_1, 1e-05);  _to_copy_64 = arg46_1 = arg47_1 = None
        getitem_147 = native_layer_norm_default_14[0]
        _to_copy_65 = torch.ops.aten._to_copy.default(arg48_1, dtype = torch.bfloat16);  arg48_1 = None
        _to_copy_66 = torch.ops.aten._to_copy.default(getitem_147, dtype = torch.bfloat16);  getitem_147 = None
        t_25 = torch.ops.aten.t.default(_to_copy_65);  _to_copy_65 = None
        view_122 = torch.ops.aten.view.default(_to_copy_66, [262144, 256]);  _to_copy_66 = None
        mm_25 = torch.ops.aten.mm.default(view_122, t_25);  view_122 = t_25 = None
        view_123 = torch.ops.aten.view.default(mm_25, [1, 512, 512, 16]);  mm_25 = None
        permute_61 = torch.ops.aten.permute.default(view_123, [0, 3, 1, 2]);  view_123 = None
        view_124 = torch.ops.aten.view.default(bitwise_and_2, [1, 1, 512, 512]);  bitwise_and_2 = None
        bitwise_not_7 = torch.ops.aten.bitwise_not.default(view_124);  view_124 = None
        masked_fill_7 = torch.ops.aten.masked_fill.Scalar(permute_61, bitwise_not_7, -10000);  permute_61 = bitwise_not_7 = None
        _to_copy_67 = torch.ops.aten._to_copy.default(getitem_144, dtype = torch.bfloat16);  getitem_144 = None
        _to_copy_68 = torch.ops.aten._to_copy.default(arg50_1, dtype = torch.bfloat16);  arg50_1 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(_to_copy_67, 3);  _to_copy_67 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(unsqueeze_31, 5);  unsqueeze_31 = None
        permute_62 = torch.ops.aten.permute.default(unsqueeze_32, [3, 0, 4, 1, 5, 2]);  unsqueeze_32 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(_to_copy_68, 4);  _to_copy_68 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(unsqueeze_33, 5);  unsqueeze_33 = None
        permute_63 = torch.ops.aten.permute.default(unsqueeze_34, [1, 4, 2, 5, 3, 0]);  unsqueeze_34 = None
        permute_64 = torch.ops.aten.permute.default(permute_62, [3, 5, 0, 1, 2, 4]);  permute_62 = None
        view_125 = torch.ops.aten.view.default(permute_64, [1, 512, 384]);  permute_64 = None
        permute_65 = torch.ops.aten.permute.default(permute_63, [5, 0, 1, 2, 4, 3]);  permute_63 = None
        view_126 = torch.ops.aten.view.default(permute_65, [1, 384, 1536]);  permute_65 = None
        bmm_6 = torch.ops.aten.bmm.default(view_125, view_126);  view_125 = view_126 = None
        view_127 = torch.ops.aten.view.default(bmm_6, [512, 1, 4, 1, 16, 24]);  bmm_6 = None
        permute_66 = torch.ops.aten.permute.default(view_127, [2, 3, 4, 0, 5, 1]);  view_127 = None
        view_128 = torch.ops.aten.view.default(permute_66, [4, 1, 16, 512, 24]);  permute_66 = None
        unbind_int_5 = torch.ops.aten.unbind.int(view_128);  view_128 = None
        getitem_150 = unbind_int_5[0]
        getitem_151 = unbind_int_5[1]
        getitem_152 = unbind_int_5[2]
        getitem_153 = unbind_int_5[3];  unbind_int_5 = None
        view_129 = torch.ops.aten.view.default(arg49_1, [1, 16, 1, 24]);  arg49_1 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_150, view_129);  getitem_150 = view_129 = None
        _to_copy_69 = torch.ops.aten._to_copy.default(add_17, dtype = torch.bfloat16);  add_17 = None
        expand_5 = torch.ops.aten.expand.default(masked_fill_7, [1, 16, 512, 512]);  masked_fill_7 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(_to_copy_69, getitem_151, getitem_152, expand_5, False);  _to_copy_69 = getitem_151 = getitem_152 = expand_5 = None
        getitem_154 = _scaled_dot_product_efficient_attention_default_5[0]
        add_18 = torch.ops.aten.add.Tensor(getitem_153, 1);  getitem_153 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(add_18);  add_18 = None
        mul_15 = torch.ops.aten.mul.Tensor(getitem_154, sigmoid_9);  getitem_154 = sigmoid_9 = None
        _to_copy_70 = torch.ops.aten._to_copy.default(arg51_1, dtype = torch.bfloat16);  arg51_1 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(mul_15, 4);  mul_15 = None
        permute_67 = torch.ops.aten.permute.default(unsqueeze_35, [0, 2, 4, 3, 1]);  unsqueeze_35 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(_to_copy_70, 3);  _to_copy_70 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, 4);  unsqueeze_36 = None
        permute_68 = torch.ops.aten.permute.default(unsqueeze_37, [3, 4, 2, 1, 0]);  unsqueeze_37 = None
        permute_69 = torch.ops.aten.permute.default(permute_67, [1, 3, 4, 0, 2]);  permute_67 = None
        clone_8 = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_8, [1, 512, 384]);  clone_8 = None
        permute_70 = torch.ops.aten.permute.default(permute_68, [3, 4, 0, 2, 1]);  permute_68 = None
        clone_9 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(clone_9, [1, 384, 384]);  clone_9 = None
        bmm_7 = torch.ops.aten.bmm.default(_unsafe_view_6, _unsafe_view_7);  _unsafe_view_6 = _unsafe_view_7 = None
        view_130 = torch.ops.aten.view.default(bmm_7, [512, 1, 1, 1, 384]);  bmm_7 = None
        permute_71 = torch.ops.aten.permute.default(view_130, [3, 0, 4, 1, 2]);  view_130 = None
        view_131 = torch.ops.aten.view.default(permute_71, [1, 512, 384]);  permute_71 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg109_1, -1)
        mul_16 = torch.ops.aten.mul.Tensor(view_131, unsqueeze_38);  view_131 = unsqueeze_38 = None
        add_19 = torch.ops.aten.add.Tensor(add_11, mul_16);  mul_16 = None
        split_tensor_25 = torch.ops.aten.split.Tensor(add_11, 512, dim = -2);  add_11 = None
        getitem_158 = split_tensor_25[0];  split_tensor_25 = None
        _to_copy_71 = torch.ops.aten._to_copy.default(getitem_158, dtype = torch.float32);  getitem_158 = None
        native_layer_norm_default_15 = torch.ops.aten.native_layer_norm.default(_to_copy_71, [384], arg40_1, arg41_1, 1e-05);  _to_copy_71 = arg40_1 = arg41_1 = None
        getitem_159 = native_layer_norm_default_15[0]
        _to_copy_72 = torch.ops.aten._to_copy.default(arg42_1, dtype = torch.bfloat16);  arg42_1 = None
        _to_copy_73 = torch.ops.aten._to_copy.default(getitem_159, dtype = torch.bfloat16);  getitem_159 = None
        t_26 = torch.ops.aten.t.default(_to_copy_72);  _to_copy_72 = None
        view_132 = torch.ops.aten.view.default(_to_copy_73, [512, 384]);  _to_copy_73 = None
        mm_26 = torch.ops.aten.mm.default(view_132, t_26);  view_132 = t_26 = None
        view_133 = torch.ops.aten.view.default(mm_26, [1, 512, 1536]);  mm_26 = None
        split_tensor_26 = torch.ops.aten.split.Tensor(view_133, 768, dim = -1);  view_133 = None
        getitem_162 = split_tensor_26[0]
        getitem_163 = split_tensor_26[1];  split_tensor_26 = None
        silu_3 = torch.ops.aten.silu.default(getitem_162);  getitem_162 = None
        mul_17 = torch.ops.aten.mul.Tensor(silu_3, getitem_163);  silu_3 = getitem_163 = None
        _to_copy_74 = torch.ops.aten._to_copy.default(arg43_1, dtype = torch.bfloat16);  arg43_1 = None
        t_27 = torch.ops.aten.t.default(_to_copy_74);  _to_copy_74 = None
        view_135 = torch.ops.aten.view.default(mul_17, [512, 768]);  mul_17 = None
        mm_27 = torch.ops.aten.mm.default(view_135, t_27);  view_135 = t_27 = None
        view_136 = torch.ops.aten.view.default(mm_27, [1, 512, 384]);  mm_27 = None
        add_20 = torch.ops.aten.add.Tensor(add_19, view_136);  add_19 = view_136 = None
        view_137 = torch.ops.aten.view.default(arg109_1, [1, 512, 1])
        view_138 = torch.ops.aten.view.default(arg109_1, [1, 1, 512])
        bitwise_and_3 = torch.ops.aten.bitwise_and.Tensor(view_137, view_138);  view_137 = view_138 = None
        _to_copy_75 = torch.ops.aten._to_copy.default(add_16, dtype = torch.float32)
        native_layer_norm_default_16 = torch.ops.aten.native_layer_norm.default(_to_copy_75, [256], arg56_1, arg57_1, 1e-05);  _to_copy_75 = arg56_1 = arg57_1 = None
        getitem_164 = native_layer_norm_default_16[0]
        split_with_sizes_default_6 = torch.ops.aten.split_with_sizes.default(arg59_1, [512, 512]);  arg59_1 = None
        getitem_167 = split_with_sizes_default_6[0]
        getitem_168 = split_with_sizes_default_6[1];  split_with_sizes_default_6 = None
        split_with_sizes_default_7 = torch.ops.aten.split_with_sizes.default(arg60_1, [512, 512, 256]);  arg60_1 = None
        getitem_169 = split_with_sizes_default_7[0]
        getitem_170 = split_with_sizes_default_7[1]
        getitem_171 = split_with_sizes_default_7[2];  split_with_sizes_default_7 = None
        _to_copy_76 = torch.ops.aten._to_copy.default(getitem_167, dtype = torch.bfloat16);  getitem_167 = None
        _to_copy_77 = torch.ops.aten._to_copy.default(getitem_164, dtype = torch.bfloat16)
        t_28 = torch.ops.aten.t.default(_to_copy_76);  _to_copy_76 = None
        view_139 = torch.ops.aten.view.default(_to_copy_77, [262144, 256]);  _to_copy_77 = None
        mm_28 = torch.ops.aten.mm.default(view_139, t_28);  view_139 = t_28 = None
        view_140 = torch.ops.aten.view.default(mm_28, [1, 512, 512, 512]);  mm_28 = None
        _to_copy_78 = torch.ops.aten._to_copy.default(getitem_169, dtype = torch.bfloat16);  getitem_169 = None
        _to_copy_79 = torch.ops.aten._to_copy.default(getitem_164, dtype = torch.bfloat16)
        t_29 = torch.ops.aten.t.default(_to_copy_78);  _to_copy_78 = None
        view_141 = torch.ops.aten.view.default(_to_copy_79, [262144, 256]);  _to_copy_79 = None
        mm_29 = torch.ops.aten.mm.default(view_141, t_29);  view_141 = t_29 = None
        view_142 = torch.ops.aten.view.default(mm_29, [1, 512, 512, 512]);  mm_29 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(view_142);  view_142 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_140, sigmoid_10);  view_140 = sigmoid_10 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(bitwise_and_3, 3)
        bitwise_not_8 = torch.ops.aten.bitwise_not.default(unsqueeze_39);  unsqueeze_39 = None
        masked_fill_8 = torch.ops.aten.masked_fill.Scalar(mul_18, bitwise_not_8, 0);  mul_18 = bitwise_not_8 = None
        split_tensor_27 = torch.ops.aten.split.Tensor(masked_fill_8, 256, dim = -1)
        getitem_174 = split_tensor_27[0]
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(getitem_174, 4);  getitem_174 = None
        permute_76 = torch.ops.aten.permute.default(unsqueeze_42, [0, 1, 4, 3, 2]);  unsqueeze_42 = None
        permute_77 = torch.ops.aten.permute.default(permute_76, [3, 1, 4, 0, 2]);  permute_76 = None
        view_145 = torch.ops.aten.view.default(permute_77, [256, 512, 512]);  permute_77 = None
        split_tensor_28 = torch.ops.aten.split.Tensor(masked_fill_8, 256, dim = -1);  masked_fill_8 = None
        getitem_177 = split_tensor_28[1];  split_tensor_28 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(getitem_177, 4);  getitem_177 = None
        permute_78 = torch.ops.aten.permute.default(unsqueeze_43, [0, 4, 1, 3, 2]);  unsqueeze_43 = None
        permute_79 = torch.ops.aten.permute.default(permute_78, [3, 4, 0, 2, 1]);  permute_78 = None
        view_146 = torch.ops.aten.view.default(permute_79, [256, 512, 512]);  permute_79 = None
        bmm_8 = torch.ops.aten.bmm.default(view_145, view_146);  view_145 = view_146 = None
        view_147 = torch.ops.aten.view.default(bmm_8, [256, 512, 1, 1, 512]);  bmm_8 = None
        permute_80 = torch.ops.aten.permute.default(view_147, [3, 1, 4, 0, 2]);  view_147 = None
        view_148 = torch.ops.aten.view.default(permute_80, [1, 512, 512, 256]);  permute_80 = None
        _to_copy_80 = torch.ops.aten._to_copy.default(getitem_168, dtype = torch.bfloat16);  getitem_168 = None
        _to_copy_81 = torch.ops.aten._to_copy.default(getitem_164, dtype = torch.bfloat16)
        t_30 = torch.ops.aten.t.default(_to_copy_80);  _to_copy_80 = None
        view_149 = torch.ops.aten.view.default(_to_copy_81, [262144, 256]);  _to_copy_81 = None
        mm_30 = torch.ops.aten.mm.default(view_149, t_30);  view_149 = t_30 = None
        view_150 = torch.ops.aten.view.default(mm_30, [1, 512, 512, 512]);  mm_30 = None
        _to_copy_82 = torch.ops.aten._to_copy.default(getitem_170, dtype = torch.bfloat16);  getitem_170 = None
        _to_copy_83 = torch.ops.aten._to_copy.default(getitem_164, dtype = torch.bfloat16)
        t_31 = torch.ops.aten.t.default(_to_copy_82);  _to_copy_82 = None
        view_151 = torch.ops.aten.view.default(_to_copy_83, [262144, 256]);  _to_copy_83 = None
        mm_31 = torch.ops.aten.mm.default(view_151, t_31);  view_151 = t_31 = None
        view_152 = torch.ops.aten.view.default(mm_31, [1, 512, 512, 512]);  mm_31 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(view_152);  view_152 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_150, sigmoid_11);  view_150 = sigmoid_11 = None
        view_153 = torch.ops.aten.view.default(mul_19, [262144, 512]);  mul_19 = None
        view_154 = torch.ops.aten.view.default(view_153, [1, 512, 512, 512]);  view_153 = None
        transpose_2 = torch.ops.aten.transpose.int(bitwise_and_3, 1, 2)
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(transpose_2, 3);  transpose_2 = None
        clone_10 = torch.ops.aten.clone.default(unsqueeze_44, memory_format = torch.contiguous_format);  unsqueeze_44 = None
        bitwise_not_9 = torch.ops.aten.bitwise_not.default(clone_10);  clone_10 = None
        masked_fill_9 = torch.ops.aten.masked_fill.Scalar(view_154, bitwise_not_9, 0);  view_154 = bitwise_not_9 = None
        view_155 = torch.ops.aten.view.default(masked_fill_9, [262144, 512]);  masked_fill_9 = None
        view_159 = torch.ops.aten.view.default(view_155, [1, 512, 512, 512])
        split_tensor_29 = torch.ops.aten.split.Tensor(view_159, 256, dim = -1);  view_159 = None
        getitem_180 = split_tensor_29[0]
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(getitem_180, 4);  getitem_180 = None
        permute_85 = torch.ops.aten.permute.default(unsqueeze_47, [0, 2, 4, 3, 1]);  unsqueeze_47 = None
        permute_86 = torch.ops.aten.permute.default(permute_85, [3, 1, 4, 0, 2]);  permute_85 = None
        view_160 = torch.ops.aten.view.default(permute_86, [256, 512, 512]);  permute_86 = None
        view_161 = torch.ops.aten.view.default(view_155, [1, 512, 512, 512]);  view_155 = None
        split_tensor_30 = torch.ops.aten.split.Tensor(view_161, 256, dim = -1);  view_161 = None
        getitem_183 = split_tensor_30[1];  split_tensor_30 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(getitem_183, 4);  getitem_183 = None
        permute_87 = torch.ops.aten.permute.default(unsqueeze_48, [0, 4, 2, 3, 1]);  unsqueeze_48 = None
        permute_88 = torch.ops.aten.permute.default(permute_87, [3, 4, 0, 2, 1]);  permute_87 = None
        view_162 = torch.ops.aten.view.default(permute_88, [256, 512, 512]);  permute_88 = None
        bmm_9 = torch.ops.aten.bmm.default(view_160, view_162);  view_160 = view_162 = None
        view_163 = torch.ops.aten.view.default(bmm_9, [256, 512, 1, 1, 512]);  bmm_9 = None
        permute_89 = torch.ops.aten.permute.default(view_163, [3, 1, 4, 0, 2]);  view_163 = None
        view_164 = torch.ops.aten.view.default(permute_89, [1, 512, 512, 256]);  permute_89 = None
        _to_copy_84 = torch.ops.aten._to_copy.default(view_148, dtype = torch.float32);  view_148 = None
        native_layer_norm_default_17 = torch.ops.aten.native_layer_norm.default(_to_copy_84, [256], None, None, 1e-05);  _to_copy_84 = None
        getitem_184 = native_layer_norm_default_17[0]
        _to_copy_85 = torch.ops.aten._to_copy.default(view_164, dtype = torch.float32);  view_164 = None
        native_layer_norm_default_18 = torch.ops.aten.native_layer_norm.default(_to_copy_85, [256], None, None, 1e-05);  _to_copy_85 = None
        getitem_187 = native_layer_norm_default_18[0]
        add_21 = torch.ops.aten.add.Tensor(getitem_184, getitem_187);  getitem_184 = getitem_187 = None
        _to_copy_86 = torch.ops.aten._to_copy.default(arg58_1, dtype = torch.bfloat16);  arg58_1 = None
        _to_copy_87 = torch.ops.aten._to_copy.default(add_21, dtype = torch.bfloat16);  add_21 = None
        t_32 = torch.ops.aten.t.default(_to_copy_86);  _to_copy_86 = None
        view_165 = torch.ops.aten.view.default(_to_copy_87, [262144, 256]);  _to_copy_87 = None
        mm_32 = torch.ops.aten.mm.default(view_165, t_32);  view_165 = t_32 = None
        view_166 = torch.ops.aten.view.default(mm_32, [1, 512, 512, 256]);  mm_32 = None
        _to_copy_88 = torch.ops.aten._to_copy.default(getitem_171, dtype = torch.bfloat16);  getitem_171 = None
        _to_copy_89 = torch.ops.aten._to_copy.default(getitem_164, dtype = torch.bfloat16);  getitem_164 = None
        t_33 = torch.ops.aten.t.default(_to_copy_88);  _to_copy_88 = None
        view_167 = torch.ops.aten.view.default(_to_copy_89, [262144, 256]);  _to_copy_89 = None
        mm_33 = torch.ops.aten.mm.default(view_167, t_33);  view_167 = t_33 = None
        view_168 = torch.ops.aten.view.default(mm_33, [1, 512, 512, 256]);  mm_33 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(view_168);  view_168 = None
        mul_20 = torch.ops.aten.mul.Tensor(view_166, sigmoid_12);  view_166 = sigmoid_12 = None
        add_22 = torch.ops.aten.add.Tensor(add_16, mul_20);  mul_20 = None
        _to_copy_90 = torch.ops.aten._to_copy.default(add_16, dtype = torch.float32)
        native_layer_norm_default_19 = torch.ops.aten.native_layer_norm.default(_to_copy_90, [256], arg61_1, arg62_1, 1e-05);  _to_copy_90 = arg61_1 = arg62_1 = None
        getitem_190 = native_layer_norm_default_19[0]
        _to_copy_91 = torch.ops.aten._to_copy.default(arg63_1, dtype = torch.bfloat16);  arg63_1 = None
        _to_copy_92 = torch.ops.aten._to_copy.default(getitem_190, dtype = torch.bfloat16);  getitem_190 = None
        t_34 = torch.ops.aten.t.default(_to_copy_91);  _to_copy_91 = None
        view_169 = torch.ops.aten.view.default(_to_copy_92, [262144, 256]);  _to_copy_92 = None
        mm_34 = torch.ops.aten.mm.default(view_169, t_34);  view_169 = t_34 = None
        view_170 = torch.ops.aten.view.default(mm_34, [1, 512, 512, 2056]);  mm_34 = None
        split_with_sizes_default_8 = torch.ops.aten.split_with_sizes.default(view_170, [2048, 8], dim = -1);  view_170 = None
        getitem_193 = split_with_sizes_default_8[0]
        getitem_194 = split_with_sizes_default_8[1];  split_with_sizes_default_8 = None
        view_171 = torch.ops.aten.view.default(getitem_194, [1, 512, 512, 2, 4]);  getitem_194 = None
        permute_90 = torch.ops.aten.permute.default(view_171, [0, 3, 4, 1, 2]);  view_171 = None
        view_172 = torch.ops.aten.view.default(permute_90, [1, 2, 4, 1, 512, 512]);  permute_90 = None
        view_173 = torch.ops.aten.view.default(bitwise_and_3, [1, 1, 1, 1, 512, 512])
        bitwise_not_10 = torch.ops.aten.bitwise_not.default(view_173);  view_173 = None
        masked_fill_10 = torch.ops.aten.masked_fill.Scalar(view_172, bitwise_not_10, -10000);  view_172 = bitwise_not_10 = None
        view_174 = torch.ops.aten.view.default(masked_fill_10, [1, 2, 4, 512, 512]);  masked_fill_10 = None
        view_175 = torch.ops.aten.view.default(view_174, [8, 1, 512, 512]);  view_174 = None
        split_tensor_31 = torch.ops.aten.split.Tensor(getitem_193, 1024, dim = -1);  getitem_193 = None
        getitem_195 = split_tensor_31[0]
        getitem_196 = split_tensor_31[1];  split_tensor_31 = None
        permute_91 = torch.ops.aten.permute.default(getitem_196, [0, 2, 1, 3]);  getitem_196 = None
        stack_2 = torch.ops.aten.stack.default([getitem_195, permute_91]);  getitem_195 = permute_91 = None
        view_176 = torch.ops.aten.view.default(stack_2, [2, 1, 512, 512, 4, 4, 64]);  stack_2 = None
        permute_92 = torch.ops.aten.permute.default(view_176, [4, 1, 0, 5, 2, 3, 6]);  view_176 = None
        clone_11 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(clone_11, [4, 8, 512, 512, 64]);  clone_11 = None
        unbind_int_6 = torch.ops.aten.unbind.int(_unsafe_view_8);  _unsafe_view_8 = None
        getitem_197 = unbind_int_6[0]
        getitem_198 = unbind_int_6[1]
        getitem_199 = unbind_int_6[2]
        getitem_200 = unbind_int_6[3];  unbind_int_6 = None
        split_tensor_32 = torch.ops.aten.split.Tensor(getitem_197, 4);  getitem_197 = None
        getitem_201 = split_tensor_32[0]
        getitem_202 = split_tensor_32[1];  split_tensor_32 = None
        split_tensor_33 = torch.ops.aten.split.Tensor(getitem_198, 4);  getitem_198 = None
        getitem_203 = split_tensor_33[0]
        getitem_204 = split_tensor_33[1];  split_tensor_33 = None
        split_tensor_34 = torch.ops.aten.split.Tensor(getitem_199, 4);  getitem_199 = None
        getitem_205 = split_tensor_34[0]
        getitem_206 = split_tensor_34[1];  split_tensor_34 = None
        split_tensor_35 = torch.ops.aten.split.Tensor(view_175, 4);  view_175 = None
        getitem_207 = split_tensor_35[0]
        getitem_208 = split_tensor_35[1];  split_tensor_35 = None
        expand_6 = torch.ops.aten.expand.default(getitem_207, [4, 512, 512, 512]);  getitem_207 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_201, getitem_203, getitem_205, expand_6, False);  getitem_201 = getitem_203 = getitem_205 = expand_6 = None
        getitem_209 = _scaled_dot_product_efficient_attention_default_6[0]
        expand_7 = torch.ops.aten.expand.default(getitem_208, [4, 512, 512, 512]);  getitem_208 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_202, getitem_204, getitem_206, expand_7, False);  getitem_202 = getitem_204 = getitem_206 = expand_7 = None
        getitem_213 = _scaled_dot_product_efficient_attention_default_7[0]
        cat_2 = torch.ops.aten.cat.default([getitem_209, getitem_213]);  getitem_209 = getitem_213 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(getitem_200);  getitem_200 = None
        mul_21 = torch.ops.aten.mul.Tensor(cat_2, sigmoid_13);  cat_2 = sigmoid_13 = None
        view_177 = torch.ops.aten.view.default(mul_21, [1, 2, 4, 512, 512, 64]);  mul_21 = None
        permute_93 = torch.ops.aten.permute.default(view_177, [0, 3, 4, 1, 2, 5]);  view_177 = None
        clone_12 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(clone_12, [1, 512, 512, 512]);  clone_12 = None
        _to_copy_93 = torch.ops.aten._to_copy.default(arg64_1, dtype = torch.bfloat16);  arg64_1 = None
        t_35 = torch.ops.aten.t.default(_to_copy_93);  _to_copy_93 = None
        view_178 = torch.ops.aten.view.default(_unsafe_view_9, [262144, 512]);  _unsafe_view_9 = None
        mm_35 = torch.ops.aten.mm.default(view_178, t_35);  view_178 = t_35 = None
        view_179 = torch.ops.aten.view.default(mm_35, [1, 512, 512, 512]);  mm_35 = None
        view_180 = torch.ops.aten.view.default(view_179, [1, 512, 512, 2, 4, 64]);  view_179 = None
        permute_94 = torch.ops.aten.permute.default(view_180, [3, 0, 1, 2, 4, 5]);  view_180 = None
        view_181 = torch.ops.aten.view.default(permute_94, [2, 1, 512, 512, 256]);  permute_94 = None
        unbind_int_7 = torch.ops.aten.unbind.int(view_181);  view_181 = None
        getitem_217 = unbind_int_7[0]
        getitem_218 = unbind_int_7[1];  unbind_int_7 = None
        permute_95 = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        permute_96 = torch.ops.aten.permute.default(permute_95, [0, 2, 1, 3]);  permute_95 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_217, permute_96);  getitem_217 = permute_96 = None
        add_24 = torch.ops.aten.add.Tensor(add_22, add_23);  add_22 = add_23 = None
        split_tensor_36 = torch.ops.aten.split.Tensor(add_16, 512, dim = -2)
        getitem_219 = split_tensor_36[0];  split_tensor_36 = None
        _to_copy_94 = torch.ops.aten._to_copy.default(getitem_219, dtype = torch.float32);  getitem_219 = None
        native_layer_norm_default_20 = torch.ops.aten.native_layer_norm.default(_to_copy_94, [256], arg52_1, arg53_1, 1e-05);  _to_copy_94 = arg52_1 = arg53_1 = None
        getitem_220 = native_layer_norm_default_20[0]
        _to_copy_95 = torch.ops.aten._to_copy.default(arg54_1, dtype = torch.bfloat16);  arg54_1 = None
        _to_copy_96 = torch.ops.aten._to_copy.default(getitem_220, dtype = torch.bfloat16);  getitem_220 = None
        t_36 = torch.ops.aten.t.default(_to_copy_95);  _to_copy_95 = None
        view_182 = torch.ops.aten.view.default(_to_copy_96, [262144, 256]);  _to_copy_96 = None
        mm_36 = torch.ops.aten.mm.default(view_182, t_36);  view_182 = t_36 = None
        view_183 = torch.ops.aten.view.default(mm_36, [1, 512, 512, 1024]);  mm_36 = None
        split_tensor_37 = torch.ops.aten.split.Tensor(view_183, 512, dim = -1);  view_183 = None
        getitem_223 = split_tensor_37[0]
        getitem_224 = split_tensor_37[1];  split_tensor_37 = None
        silu_4 = torch.ops.aten.silu.default(getitem_223);  getitem_223 = None
        mul_22 = torch.ops.aten.mul.Tensor(silu_4, getitem_224);  silu_4 = getitem_224 = None
        _to_copy_97 = torch.ops.aten._to_copy.default(arg55_1, dtype = torch.bfloat16);  arg55_1 = None
        t_37 = torch.ops.aten.t.default(_to_copy_97);  _to_copy_97 = None
        view_185 = torch.ops.aten.view.default(mul_22, [262144, 512]);  mul_22 = None
        mm_37 = torch.ops.aten.mm.default(view_185, t_37);  view_185 = t_37 = None
        view_186 = torch.ops.aten.view.default(mm_37, [1, 512, 512, 256]);  mm_37 = None
        add_25 = torch.ops.aten.add.Tensor(add_24, view_186);  add_24 = view_186 = None
        _to_copy_98 = torch.ops.aten._to_copy.default(add_20, dtype = torch.float32)
        native_layer_norm_default_21 = torch.ops.aten.native_layer_norm.default(_to_copy_98, [384], arg69_1, arg70_1, 1e-05);  _to_copy_98 = arg69_1 = arg70_1 = None
        getitem_225 = native_layer_norm_default_21[0]
        _to_copy_99 = torch.ops.aten._to_copy.default(add_16, dtype = torch.float32);  add_16 = None
        native_layer_norm_default_22 = torch.ops.aten.native_layer_norm.default(_to_copy_99, [256], arg71_1, arg72_1, 1e-05);  _to_copy_99 = arg71_1 = arg72_1 = None
        getitem_228 = native_layer_norm_default_22[0]
        _to_copy_100 = torch.ops.aten._to_copy.default(arg73_1, dtype = torch.bfloat16);  arg73_1 = None
        _to_copy_101 = torch.ops.aten._to_copy.default(getitem_228, dtype = torch.bfloat16);  getitem_228 = None
        t_38 = torch.ops.aten.t.default(_to_copy_100);  _to_copy_100 = None
        view_187 = torch.ops.aten.view.default(_to_copy_101, [262144, 256]);  _to_copy_101 = None
        mm_38 = torch.ops.aten.mm.default(view_187, t_38);  view_187 = t_38 = None
        view_188 = torch.ops.aten.view.default(mm_38, [1, 512, 512, 16]);  mm_38 = None
        permute_97 = torch.ops.aten.permute.default(view_188, [0, 3, 1, 2]);  view_188 = None
        view_189 = torch.ops.aten.view.default(bitwise_and_3, [1, 1, 512, 512]);  bitwise_and_3 = None
        bitwise_not_11 = torch.ops.aten.bitwise_not.default(view_189);  view_189 = None
        masked_fill_11 = torch.ops.aten.masked_fill.Scalar(permute_97, bitwise_not_11, -10000);  permute_97 = bitwise_not_11 = None
        _to_copy_102 = torch.ops.aten._to_copy.default(getitem_225, dtype = torch.bfloat16);  getitem_225 = None
        _to_copy_103 = torch.ops.aten._to_copy.default(arg75_1, dtype = torch.bfloat16);  arg75_1 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(_to_copy_102, 3);  _to_copy_102 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(unsqueeze_49, 4);  unsqueeze_49 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, 5);  unsqueeze_50 = None
        permute_98 = torch.ops.aten.permute.default(unsqueeze_51, [3, 0, 4, 1, 5, 2]);  unsqueeze_51 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(_to_copy_103, 4);  _to_copy_103 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, 5);  unsqueeze_52 = None
        permute_99 = torch.ops.aten.permute.default(unsqueeze_53, [1, 4, 2, 5, 3, 0]);  unsqueeze_53 = None
        permute_100 = torch.ops.aten.permute.default(permute_98, [3, 5, 0, 1, 2, 4]);  permute_98 = None
        view_190 = torch.ops.aten.view.default(permute_100, [1, 512, 384]);  permute_100 = None
        permute_101 = torch.ops.aten.permute.default(permute_99, [5, 0, 1, 2, 4, 3]);  permute_99 = None
        view_191 = torch.ops.aten.view.default(permute_101, [1, 384, 1536]);  permute_101 = None
        bmm_10 = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
        view_192 = torch.ops.aten.view.default(bmm_10, [512, 1, 4, 1, 16, 24]);  bmm_10 = None
        permute_102 = torch.ops.aten.permute.default(view_192, [2, 3, 4, 0, 5, 1]);  view_192 = None
        view_193 = torch.ops.aten.view.default(permute_102, [4, 1, 16, 512, 24]);  permute_102 = None
        unbind_int_8 = torch.ops.aten.unbind.int(view_193);  view_193 = None
        getitem_231 = unbind_int_8[0]
        getitem_232 = unbind_int_8[1]
        getitem_233 = unbind_int_8[2]
        getitem_234 = unbind_int_8[3];  unbind_int_8 = None
        view_194 = torch.ops.aten.view.default(arg74_1, [1, 16, 1, 24]);  arg74_1 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_231, view_194);  getitem_231 = view_194 = None
        _to_copy_104 = torch.ops.aten._to_copy.default(add_26, dtype = torch.bfloat16);  add_26 = None
        expand_8 = torch.ops.aten.expand.default(masked_fill_11, [1, 16, 512, 512]);  masked_fill_11 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(_to_copy_104, getitem_232, getitem_233, expand_8, False);  _to_copy_104 = getitem_232 = getitem_233 = expand_8 = None
        getitem_235 = _scaled_dot_product_efficient_attention_default_8[0]
        add_27 = torch.ops.aten.add.Tensor(getitem_234, 1);  getitem_234 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(add_27);  add_27 = None
        mul_23 = torch.ops.aten.mul.Tensor(getitem_235, sigmoid_14);  getitem_235 = sigmoid_14 = None
        _to_copy_105 = torch.ops.aten._to_copy.default(arg76_1, dtype = torch.bfloat16);  arg76_1 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(mul_23, 4);  mul_23 = None
        permute_103 = torch.ops.aten.permute.default(unsqueeze_54, [0, 2, 4, 3, 1]);  unsqueeze_54 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(_to_copy_105, 3);  _to_copy_105 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        permute_104 = torch.ops.aten.permute.default(unsqueeze_56, [3, 4, 2, 1, 0]);  unsqueeze_56 = None
        permute_105 = torch.ops.aten.permute.default(permute_103, [1, 3, 4, 0, 2]);  permute_103 = None
        clone_13 = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(clone_13, [1, 512, 384]);  clone_13 = None
        permute_106 = torch.ops.aten.permute.default(permute_104, [3, 4, 0, 2, 1]);  permute_104 = None
        clone_14 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        _unsafe_view_11 = torch.ops.aten._unsafe_view.default(clone_14, [1, 384, 384]);  clone_14 = None
        bmm_11 = torch.ops.aten.bmm.default(_unsafe_view_10, _unsafe_view_11);  _unsafe_view_10 = _unsafe_view_11 = None
        view_195 = torch.ops.aten.view.default(bmm_11, [512, 1, 1, 1, 384]);  bmm_11 = None
        permute_107 = torch.ops.aten.permute.default(view_195, [3, 0, 4, 1, 2]);  view_195 = None
        view_196 = torch.ops.aten.view.default(permute_107, [1, 512, 384]);  permute_107 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(arg109_1, -1)
        mul_24 = torch.ops.aten.mul.Tensor(view_196, unsqueeze_57);  view_196 = unsqueeze_57 = None
        add_28 = torch.ops.aten.add.Tensor(add_20, mul_24);  mul_24 = None
        split_tensor_38 = torch.ops.aten.split.Tensor(add_20, 512, dim = -2);  add_20 = None
        getitem_239 = split_tensor_38[0];  split_tensor_38 = None
        _to_copy_106 = torch.ops.aten._to_copy.default(getitem_239, dtype = torch.float32);  getitem_239 = None
        native_layer_norm_default_23 = torch.ops.aten.native_layer_norm.default(_to_copy_106, [384], arg65_1, arg66_1, 1e-05);  _to_copy_106 = arg65_1 = arg66_1 = None
        getitem_240 = native_layer_norm_default_23[0]
        _to_copy_107 = torch.ops.aten._to_copy.default(arg67_1, dtype = torch.bfloat16);  arg67_1 = None
        _to_copy_108 = torch.ops.aten._to_copy.default(getitem_240, dtype = torch.bfloat16);  getitem_240 = None
        t_39 = torch.ops.aten.t.default(_to_copy_107);  _to_copy_107 = None
        view_197 = torch.ops.aten.view.default(_to_copy_108, [512, 384]);  _to_copy_108 = None
        mm_39 = torch.ops.aten.mm.default(view_197, t_39);  view_197 = t_39 = None
        view_198 = torch.ops.aten.view.default(mm_39, [1, 512, 1536]);  mm_39 = None
        split_tensor_39 = torch.ops.aten.split.Tensor(view_198, 768, dim = -1);  view_198 = None
        getitem_243 = split_tensor_39[0]
        getitem_244 = split_tensor_39[1];  split_tensor_39 = None
        silu_5 = torch.ops.aten.silu.default(getitem_243);  getitem_243 = None
        mul_25 = torch.ops.aten.mul.Tensor(silu_5, getitem_244);  silu_5 = getitem_244 = None
        _to_copy_109 = torch.ops.aten._to_copy.default(arg68_1, dtype = torch.bfloat16);  arg68_1 = None
        t_40 = torch.ops.aten.t.default(_to_copy_109);  _to_copy_109 = None
        view_200 = torch.ops.aten.view.default(mul_25, [512, 768]);  mul_25 = None
        mm_40 = torch.ops.aten.mm.default(view_200, t_40);  view_200 = t_40 = None
        view_201 = torch.ops.aten.view.default(mm_40, [1, 512, 384]);  mm_40 = None
        add_29 = torch.ops.aten.add.Tensor(add_28, view_201);  add_28 = view_201 = None
        view_202 = torch.ops.aten.view.default(arg109_1, [1, 512, 1])
        view_203 = torch.ops.aten.view.default(arg109_1, [1, 1, 512])
        bitwise_and_4 = torch.ops.aten.bitwise_and.Tensor(view_202, view_203);  view_202 = view_203 = None
        _to_copy_110 = torch.ops.aten._to_copy.default(add_25, dtype = torch.float32)
        native_layer_norm_default_24 = torch.ops.aten.native_layer_norm.default(_to_copy_110, [256], arg81_1, arg82_1, 1e-05);  _to_copy_110 = arg81_1 = arg82_1 = None
        getitem_245 = native_layer_norm_default_24[0]
        split_with_sizes_default_9 = torch.ops.aten.split_with_sizes.default(arg84_1, [512, 512]);  arg84_1 = None
        getitem_248 = split_with_sizes_default_9[0]
        getitem_249 = split_with_sizes_default_9[1];  split_with_sizes_default_9 = None
        split_with_sizes_default_10 = torch.ops.aten.split_with_sizes.default(arg85_1, [512, 512, 256]);  arg85_1 = None
        getitem_250 = split_with_sizes_default_10[0]
        getitem_251 = split_with_sizes_default_10[1]
        getitem_252 = split_with_sizes_default_10[2];  split_with_sizes_default_10 = None
        _to_copy_111 = torch.ops.aten._to_copy.default(getitem_248, dtype = torch.bfloat16);  getitem_248 = None
        _to_copy_112 = torch.ops.aten._to_copy.default(getitem_245, dtype = torch.bfloat16)
        t_41 = torch.ops.aten.t.default(_to_copy_111);  _to_copy_111 = None
        view_204 = torch.ops.aten.view.default(_to_copy_112, [262144, 256]);  _to_copy_112 = None
        mm_41 = torch.ops.aten.mm.default(view_204, t_41);  view_204 = t_41 = None
        view_205 = torch.ops.aten.view.default(mm_41, [1, 512, 512, 512]);  mm_41 = None
        _to_copy_113 = torch.ops.aten._to_copy.default(getitem_250, dtype = torch.bfloat16);  getitem_250 = None
        _to_copy_114 = torch.ops.aten._to_copy.default(getitem_245, dtype = torch.bfloat16)
        t_42 = torch.ops.aten.t.default(_to_copy_113);  _to_copy_113 = None
        view_206 = torch.ops.aten.view.default(_to_copy_114, [262144, 256]);  _to_copy_114 = None
        mm_42 = torch.ops.aten.mm.default(view_206, t_42);  view_206 = t_42 = None
        view_207 = torch.ops.aten.view.default(mm_42, [1, 512, 512, 512]);  mm_42 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(view_207);  view_207 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_205, sigmoid_15);  view_205 = sigmoid_15 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(bitwise_and_4, 3)
        bitwise_not_12 = torch.ops.aten.bitwise_not.default(unsqueeze_58);  unsqueeze_58 = None
        masked_fill_12 = torch.ops.aten.masked_fill.Scalar(mul_26, bitwise_not_12, 0);  mul_26 = bitwise_not_12 = None
        split_tensor_40 = torch.ops.aten.split.Tensor(masked_fill_12, 256, dim = -1)
        getitem_255 = split_tensor_40[0]
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(getitem_255, 4);  getitem_255 = None
        permute_112 = torch.ops.aten.permute.default(unsqueeze_61, [0, 1, 4, 3, 2]);  unsqueeze_61 = None
        permute_113 = torch.ops.aten.permute.default(permute_112, [3, 1, 4, 0, 2]);  permute_112 = None
        view_210 = torch.ops.aten.view.default(permute_113, [256, 512, 512]);  permute_113 = None
        split_tensor_41 = torch.ops.aten.split.Tensor(masked_fill_12, 256, dim = -1);  masked_fill_12 = None
        getitem_258 = split_tensor_41[1];  split_tensor_41 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(getitem_258, 4);  getitem_258 = None
        permute_114 = torch.ops.aten.permute.default(unsqueeze_62, [0, 4, 1, 3, 2]);  unsqueeze_62 = None
        permute_115 = torch.ops.aten.permute.default(permute_114, [3, 4, 0, 2, 1]);  permute_114 = None
        view_211 = torch.ops.aten.view.default(permute_115, [256, 512, 512]);  permute_115 = None
        bmm_12 = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
        view_212 = torch.ops.aten.view.default(bmm_12, [256, 512, 1, 1, 512]);  bmm_12 = None
        permute_116 = torch.ops.aten.permute.default(view_212, [3, 1, 4, 0, 2]);  view_212 = None
        view_213 = torch.ops.aten.view.default(permute_116, [1, 512, 512, 256]);  permute_116 = None
        _to_copy_115 = torch.ops.aten._to_copy.default(getitem_249, dtype = torch.bfloat16);  getitem_249 = None
        _to_copy_116 = torch.ops.aten._to_copy.default(getitem_245, dtype = torch.bfloat16)
        t_43 = torch.ops.aten.t.default(_to_copy_115);  _to_copy_115 = None
        view_214 = torch.ops.aten.view.default(_to_copy_116, [262144, 256]);  _to_copy_116 = None
        mm_43 = torch.ops.aten.mm.default(view_214, t_43);  view_214 = t_43 = None
        view_215 = torch.ops.aten.view.default(mm_43, [1, 512, 512, 512]);  mm_43 = None
        _to_copy_117 = torch.ops.aten._to_copy.default(getitem_251, dtype = torch.bfloat16);  getitem_251 = None
        _to_copy_118 = torch.ops.aten._to_copy.default(getitem_245, dtype = torch.bfloat16)
        t_44 = torch.ops.aten.t.default(_to_copy_117);  _to_copy_117 = None
        view_216 = torch.ops.aten.view.default(_to_copy_118, [262144, 256]);  _to_copy_118 = None
        mm_44 = torch.ops.aten.mm.default(view_216, t_44);  view_216 = t_44 = None
        view_217 = torch.ops.aten.view.default(mm_44, [1, 512, 512, 512]);  mm_44 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(view_217);  view_217 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_215, sigmoid_16);  view_215 = sigmoid_16 = None
        view_218 = torch.ops.aten.view.default(mul_27, [262144, 512]);  mul_27 = None
        view_219 = torch.ops.aten.view.default(view_218, [1, 512, 512, 512]);  view_218 = None
        transpose_3 = torch.ops.aten.transpose.int(bitwise_and_4, 1, 2)
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(transpose_3, 3);  transpose_3 = None
        clone_15 = torch.ops.aten.clone.default(unsqueeze_63, memory_format = torch.contiguous_format);  unsqueeze_63 = None
        bitwise_not_13 = torch.ops.aten.bitwise_not.default(clone_15);  clone_15 = None
        masked_fill_13 = torch.ops.aten.masked_fill.Scalar(view_219, bitwise_not_13, 0);  view_219 = bitwise_not_13 = None
        view_220 = torch.ops.aten.view.default(masked_fill_13, [262144, 512]);  masked_fill_13 = None
        view_224 = torch.ops.aten.view.default(view_220, [1, 512, 512, 512])
        split_tensor_42 = torch.ops.aten.split.Tensor(view_224, 256, dim = -1);  view_224 = None
        getitem_261 = split_tensor_42[0]
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(getitem_261, 4);  getitem_261 = None
        permute_121 = torch.ops.aten.permute.default(unsqueeze_66, [0, 2, 4, 3, 1]);  unsqueeze_66 = None
        permute_122 = torch.ops.aten.permute.default(permute_121, [3, 1, 4, 0, 2]);  permute_121 = None
        view_225 = torch.ops.aten.view.default(permute_122, [256, 512, 512]);  permute_122 = None
        view_226 = torch.ops.aten.view.default(view_220, [1, 512, 512, 512]);  view_220 = None
        split_tensor_43 = torch.ops.aten.split.Tensor(view_226, 256, dim = -1);  view_226 = None
        getitem_264 = split_tensor_43[1];  split_tensor_43 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(getitem_264, 4);  getitem_264 = None
        permute_123 = torch.ops.aten.permute.default(unsqueeze_67, [0, 4, 2, 3, 1]);  unsqueeze_67 = None
        permute_124 = torch.ops.aten.permute.default(permute_123, [3, 4, 0, 2, 1]);  permute_123 = None
        view_227 = torch.ops.aten.view.default(permute_124, [256, 512, 512]);  permute_124 = None
        bmm_13 = torch.ops.aten.bmm.default(view_225, view_227);  view_225 = view_227 = None
        view_228 = torch.ops.aten.view.default(bmm_13, [256, 512, 1, 1, 512]);  bmm_13 = None
        permute_125 = torch.ops.aten.permute.default(view_228, [3, 1, 4, 0, 2]);  view_228 = None
        view_229 = torch.ops.aten.view.default(permute_125, [1, 512, 512, 256]);  permute_125 = None
        _to_copy_119 = torch.ops.aten._to_copy.default(view_213, dtype = torch.float32);  view_213 = None
        native_layer_norm_default_25 = torch.ops.aten.native_layer_norm.default(_to_copy_119, [256], None, None, 1e-05);  _to_copy_119 = None
        getitem_265 = native_layer_norm_default_25[0]
        _to_copy_120 = torch.ops.aten._to_copy.default(view_229, dtype = torch.float32);  view_229 = None
        native_layer_norm_default_26 = torch.ops.aten.native_layer_norm.default(_to_copy_120, [256], None, None, 1e-05);  _to_copy_120 = None
        getitem_268 = native_layer_norm_default_26[0]
        add_30 = torch.ops.aten.add.Tensor(getitem_265, getitem_268);  getitem_265 = getitem_268 = None
        _to_copy_121 = torch.ops.aten._to_copy.default(arg83_1, dtype = torch.bfloat16);  arg83_1 = None
        _to_copy_122 = torch.ops.aten._to_copy.default(add_30, dtype = torch.bfloat16);  add_30 = None
        t_45 = torch.ops.aten.t.default(_to_copy_121);  _to_copy_121 = None
        view_230 = torch.ops.aten.view.default(_to_copy_122, [262144, 256]);  _to_copy_122 = None
        mm_45 = torch.ops.aten.mm.default(view_230, t_45);  view_230 = t_45 = None
        view_231 = torch.ops.aten.view.default(mm_45, [1, 512, 512, 256]);  mm_45 = None
        _to_copy_123 = torch.ops.aten._to_copy.default(getitem_252, dtype = torch.bfloat16);  getitem_252 = None
        _to_copy_124 = torch.ops.aten._to_copy.default(getitem_245, dtype = torch.bfloat16);  getitem_245 = None
        t_46 = torch.ops.aten.t.default(_to_copy_123);  _to_copy_123 = None
        view_232 = torch.ops.aten.view.default(_to_copy_124, [262144, 256]);  _to_copy_124 = None
        mm_46 = torch.ops.aten.mm.default(view_232, t_46);  view_232 = t_46 = None
        view_233 = torch.ops.aten.view.default(mm_46, [1, 512, 512, 256]);  mm_46 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(view_233);  view_233 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_231, sigmoid_17);  view_231 = sigmoid_17 = None
        add_31 = torch.ops.aten.add.Tensor(add_25, mul_28);  mul_28 = None
        _to_copy_125 = torch.ops.aten._to_copy.default(add_25, dtype = torch.float32)
        native_layer_norm_default_27 = torch.ops.aten.native_layer_norm.default(_to_copy_125, [256], arg86_1, arg87_1, 1e-05);  _to_copy_125 = arg86_1 = arg87_1 = None
        getitem_271 = native_layer_norm_default_27[0]
        _to_copy_126 = torch.ops.aten._to_copy.default(arg88_1, dtype = torch.bfloat16);  arg88_1 = None
        _to_copy_127 = torch.ops.aten._to_copy.default(getitem_271, dtype = torch.bfloat16);  getitem_271 = None
        t_47 = torch.ops.aten.t.default(_to_copy_126);  _to_copy_126 = None
        view_234 = torch.ops.aten.view.default(_to_copy_127, [262144, 256]);  _to_copy_127 = None
        mm_47 = torch.ops.aten.mm.default(view_234, t_47);  view_234 = t_47 = None
        view_235 = torch.ops.aten.view.default(mm_47, [1, 512, 512, 2056]);  mm_47 = None
        split_with_sizes_default_11 = torch.ops.aten.split_with_sizes.default(view_235, [2048, 8], dim = -1);  view_235 = None
        getitem_274 = split_with_sizes_default_11[0]
        getitem_275 = split_with_sizes_default_11[1];  split_with_sizes_default_11 = None
        view_236 = torch.ops.aten.view.default(getitem_275, [1, 512, 512, 2, 4]);  getitem_275 = None
        permute_126 = torch.ops.aten.permute.default(view_236, [0, 3, 4, 1, 2]);  view_236 = None
        view_237 = torch.ops.aten.view.default(permute_126, [1, 2, 4, 1, 512, 512]);  permute_126 = None
        view_238 = torch.ops.aten.view.default(bitwise_and_4, [1, 1, 1, 1, 512, 512])
        bitwise_not_14 = torch.ops.aten.bitwise_not.default(view_238);  view_238 = None
        masked_fill_14 = torch.ops.aten.masked_fill.Scalar(view_237, bitwise_not_14, -10000);  view_237 = bitwise_not_14 = None
        view_239 = torch.ops.aten.view.default(masked_fill_14, [1, 2, 4, 512, 512]);  masked_fill_14 = None
        view_240 = torch.ops.aten.view.default(view_239, [8, 1, 512, 512]);  view_239 = None
        split_tensor_44 = torch.ops.aten.split.Tensor(getitem_274, 1024, dim = -1);  getitem_274 = None
        getitem_276 = split_tensor_44[0]
        getitem_277 = split_tensor_44[1];  split_tensor_44 = None
        permute_127 = torch.ops.aten.permute.default(getitem_277, [0, 2, 1, 3]);  getitem_277 = None
        stack_3 = torch.ops.aten.stack.default([getitem_276, permute_127]);  getitem_276 = permute_127 = None
        view_241 = torch.ops.aten.view.default(stack_3, [2, 1, 512, 512, 4, 4, 64]);  stack_3 = None
        permute_128 = torch.ops.aten.permute.default(view_241, [4, 1, 0, 5, 2, 3, 6]);  view_241 = None
        clone_16 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        _unsafe_view_12 = torch.ops.aten._unsafe_view.default(clone_16, [4, 8, 512, 512, 64]);  clone_16 = None
        unbind_int_9 = torch.ops.aten.unbind.int(_unsafe_view_12);  _unsafe_view_12 = None
        getitem_278 = unbind_int_9[0]
        getitem_279 = unbind_int_9[1]
        getitem_280 = unbind_int_9[2]
        getitem_281 = unbind_int_9[3];  unbind_int_9 = None
        split_tensor_45 = torch.ops.aten.split.Tensor(getitem_278, 4);  getitem_278 = None
        getitem_282 = split_tensor_45[0]
        getitem_283 = split_tensor_45[1];  split_tensor_45 = None
        split_tensor_46 = torch.ops.aten.split.Tensor(getitem_279, 4);  getitem_279 = None
        getitem_284 = split_tensor_46[0]
        getitem_285 = split_tensor_46[1];  split_tensor_46 = None
        split_tensor_47 = torch.ops.aten.split.Tensor(getitem_280, 4);  getitem_280 = None
        getitem_286 = split_tensor_47[0]
        getitem_287 = split_tensor_47[1];  split_tensor_47 = None
        split_tensor_48 = torch.ops.aten.split.Tensor(view_240, 4);  view_240 = None
        getitem_288 = split_tensor_48[0]
        getitem_289 = split_tensor_48[1];  split_tensor_48 = None
        expand_9 = torch.ops.aten.expand.default(getitem_288, [4, 512, 512, 512]);  getitem_288 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_282, getitem_284, getitem_286, expand_9, False);  getitem_282 = getitem_284 = getitem_286 = expand_9 = None
        getitem_290 = _scaled_dot_product_efficient_attention_default_9[0]
        expand_10 = torch.ops.aten.expand.default(getitem_289, [4, 512, 512, 512]);  getitem_289 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_283, getitem_285, getitem_287, expand_10, False);  getitem_283 = getitem_285 = getitem_287 = expand_10 = None
        getitem_294 = _scaled_dot_product_efficient_attention_default_10[0]
        cat_3 = torch.ops.aten.cat.default([getitem_290, getitem_294]);  getitem_290 = getitem_294 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(getitem_281);  getitem_281 = None
        mul_29 = torch.ops.aten.mul.Tensor(cat_3, sigmoid_18);  cat_3 = sigmoid_18 = None
        view_242 = torch.ops.aten.view.default(mul_29, [1, 2, 4, 512, 512, 64]);  mul_29 = None
        permute_129 = torch.ops.aten.permute.default(view_242, [0, 3, 4, 1, 2, 5]);  view_242 = None
        clone_17 = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        _unsafe_view_13 = torch.ops.aten._unsafe_view.default(clone_17, [1, 512, 512, 512]);  clone_17 = None
        _to_copy_128 = torch.ops.aten._to_copy.default(arg89_1, dtype = torch.bfloat16);  arg89_1 = None
        t_48 = torch.ops.aten.t.default(_to_copy_128);  _to_copy_128 = None
        view_243 = torch.ops.aten.view.default(_unsafe_view_13, [262144, 512]);  _unsafe_view_13 = None
        mm_48 = torch.ops.aten.mm.default(view_243, t_48);  view_243 = t_48 = None
        view_244 = torch.ops.aten.view.default(mm_48, [1, 512, 512, 512]);  mm_48 = None
        view_245 = torch.ops.aten.view.default(view_244, [1, 512, 512, 2, 4, 64]);  view_244 = None
        permute_130 = torch.ops.aten.permute.default(view_245, [3, 0, 1, 2, 4, 5]);  view_245 = None
        view_246 = torch.ops.aten.view.default(permute_130, [2, 1, 512, 512, 256]);  permute_130 = None
        unbind_int_10 = torch.ops.aten.unbind.int(view_246);  view_246 = None
        getitem_298 = unbind_int_10[0]
        getitem_299 = unbind_int_10[1];  unbind_int_10 = None
        permute_131 = torch.ops.aten.permute.default(getitem_299, [0, 2, 1, 3]);  getitem_299 = None
        permute_132 = torch.ops.aten.permute.default(permute_131, [0, 2, 1, 3]);  permute_131 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_298, permute_132);  getitem_298 = permute_132 = None
        add_33 = torch.ops.aten.add.Tensor(add_31, add_32);  add_31 = add_32 = None
        split_tensor_49 = torch.ops.aten.split.Tensor(add_25, 512, dim = -2)
        getitem_300 = split_tensor_49[0];  split_tensor_49 = None
        _to_copy_129 = torch.ops.aten._to_copy.default(getitem_300, dtype = torch.float32);  getitem_300 = None
        native_layer_norm_default_28 = torch.ops.aten.native_layer_norm.default(_to_copy_129, [256], arg77_1, arg78_1, 1e-05);  _to_copy_129 = arg77_1 = arg78_1 = None
        getitem_301 = native_layer_norm_default_28[0]
        _to_copy_130 = torch.ops.aten._to_copy.default(arg79_1, dtype = torch.bfloat16);  arg79_1 = None
        _to_copy_131 = torch.ops.aten._to_copy.default(getitem_301, dtype = torch.bfloat16);  getitem_301 = None
        t_49 = torch.ops.aten.t.default(_to_copy_130);  _to_copy_130 = None
        view_247 = torch.ops.aten.view.default(_to_copy_131, [262144, 256]);  _to_copy_131 = None
        mm_49 = torch.ops.aten.mm.default(view_247, t_49);  view_247 = t_49 = None
        view_248 = torch.ops.aten.view.default(mm_49, [1, 512, 512, 1024]);  mm_49 = None
        split_tensor_50 = torch.ops.aten.split.Tensor(view_248, 512, dim = -1);  view_248 = None
        getitem_304 = split_tensor_50[0]
        getitem_305 = split_tensor_50[1];  split_tensor_50 = None
        silu_6 = torch.ops.aten.silu.default(getitem_304);  getitem_304 = None
        mul_30 = torch.ops.aten.mul.Tensor(silu_6, getitem_305);  silu_6 = getitem_305 = None
        _to_copy_132 = torch.ops.aten._to_copy.default(arg80_1, dtype = torch.bfloat16);  arg80_1 = None
        t_50 = torch.ops.aten.t.default(_to_copy_132);  _to_copy_132 = None
        view_250 = torch.ops.aten.view.default(mul_30, [262144, 512]);  mul_30 = None
        mm_50 = torch.ops.aten.mm.default(view_250, t_50);  view_250 = t_50 = None
        view_251 = torch.ops.aten.view.default(mm_50, [1, 512, 512, 256]);  mm_50 = None
        add_34 = torch.ops.aten.add.Tensor(add_33, view_251);  add_33 = view_251 = None
        _to_copy_133 = torch.ops.aten._to_copy.default(add_29, dtype = torch.float32)
        native_layer_norm_default_29 = torch.ops.aten.native_layer_norm.default(_to_copy_133, [384], arg94_1, arg95_1, 1e-05);  _to_copy_133 = arg94_1 = arg95_1 = None
        getitem_306 = native_layer_norm_default_29[0]
        _to_copy_134 = torch.ops.aten._to_copy.default(add_25, dtype = torch.float32);  add_25 = None
        native_layer_norm_default_30 = torch.ops.aten.native_layer_norm.default(_to_copy_134, [256], arg96_1, arg97_1, 1e-05);  _to_copy_134 = arg96_1 = arg97_1 = None
        getitem_309 = native_layer_norm_default_30[0]
        _to_copy_135 = torch.ops.aten._to_copy.default(arg98_1, dtype = torch.bfloat16);  arg98_1 = None
        _to_copy_136 = torch.ops.aten._to_copy.default(getitem_309, dtype = torch.bfloat16);  getitem_309 = None
        t_51 = torch.ops.aten.t.default(_to_copy_135);  _to_copy_135 = None
        view_252 = torch.ops.aten.view.default(_to_copy_136, [262144, 256]);  _to_copy_136 = None
        mm_51 = torch.ops.aten.mm.default(view_252, t_51);  view_252 = t_51 = None
        view_253 = torch.ops.aten.view.default(mm_51, [1, 512, 512, 16]);  mm_51 = None
        permute_133 = torch.ops.aten.permute.default(view_253, [0, 3, 1, 2]);  view_253 = None
        view_254 = torch.ops.aten.view.default(bitwise_and_4, [1, 1, 512, 512]);  bitwise_and_4 = None
        bitwise_not_15 = torch.ops.aten.bitwise_not.default(view_254);  view_254 = None
        masked_fill_15 = torch.ops.aten.masked_fill.Scalar(permute_133, bitwise_not_15, -10000);  permute_133 = bitwise_not_15 = None
        _to_copy_137 = torch.ops.aten._to_copy.default(getitem_306, dtype = torch.bfloat16);  getitem_306 = None
        _to_copy_138 = torch.ops.aten._to_copy.default(arg100_1, dtype = torch.bfloat16);  arg100_1 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(_to_copy_137, 3);  _to_copy_137 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, 4);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(unsqueeze_69, 5);  unsqueeze_69 = None
        permute_134 = torch.ops.aten.permute.default(unsqueeze_70, [3, 0, 4, 1, 5, 2]);  unsqueeze_70 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(_to_copy_138, 4);  _to_copy_138 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(unsqueeze_71, 5);  unsqueeze_71 = None
        permute_135 = torch.ops.aten.permute.default(unsqueeze_72, [1, 4, 2, 5, 3, 0]);  unsqueeze_72 = None
        permute_136 = torch.ops.aten.permute.default(permute_134, [3, 5, 0, 1, 2, 4]);  permute_134 = None
        view_255 = torch.ops.aten.view.default(permute_136, [1, 512, 384]);  permute_136 = None
        permute_137 = torch.ops.aten.permute.default(permute_135, [5, 0, 1, 2, 4, 3]);  permute_135 = None
        view_256 = torch.ops.aten.view.default(permute_137, [1, 384, 1536]);  permute_137 = None
        bmm_14 = torch.ops.aten.bmm.default(view_255, view_256);  view_255 = view_256 = None
        view_257 = torch.ops.aten.view.default(bmm_14, [512, 1, 4, 1, 16, 24]);  bmm_14 = None
        permute_138 = torch.ops.aten.permute.default(view_257, [2, 3, 4, 0, 5, 1]);  view_257 = None
        view_258 = torch.ops.aten.view.default(permute_138, [4, 1, 16, 512, 24]);  permute_138 = None
        unbind_int_11 = torch.ops.aten.unbind.int(view_258);  view_258 = None
        getitem_312 = unbind_int_11[0]
        getitem_313 = unbind_int_11[1]
        getitem_314 = unbind_int_11[2]
        getitem_315 = unbind_int_11[3];  unbind_int_11 = None
        view_259 = torch.ops.aten.view.default(arg99_1, [1, 16, 1, 24]);  arg99_1 = None
        add_35 = torch.ops.aten.add.Tensor(getitem_312, view_259);  getitem_312 = view_259 = None
        _to_copy_139 = torch.ops.aten._to_copy.default(add_35, dtype = torch.bfloat16);  add_35 = None
        expand_11 = torch.ops.aten.expand.default(masked_fill_15, [1, 16, 512, 512]);  masked_fill_15 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(_to_copy_139, getitem_313, getitem_314, expand_11, False);  _to_copy_139 = getitem_313 = getitem_314 = expand_11 = None
        getitem_316 = _scaled_dot_product_efficient_attention_default_11[0]
        add_36 = torch.ops.aten.add.Tensor(getitem_315, 1);  getitem_315 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(add_36);  add_36 = None
        mul_31 = torch.ops.aten.mul.Tensor(getitem_316, sigmoid_19);  getitem_316 = sigmoid_19 = None
        _to_copy_140 = torch.ops.aten._to_copy.default(arg101_1, dtype = torch.bfloat16);  arg101_1 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(mul_31, 4);  mul_31 = None
        permute_139 = torch.ops.aten.permute.default(unsqueeze_73, [0, 2, 4, 3, 1]);  unsqueeze_73 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(_to_copy_140, 3);  _to_copy_140 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, 4);  unsqueeze_74 = None
        permute_140 = torch.ops.aten.permute.default(unsqueeze_75, [3, 4, 2, 1, 0]);  unsqueeze_75 = None
        permute_141 = torch.ops.aten.permute.default(permute_139, [1, 3, 4, 0, 2]);  permute_139 = None
        clone_18 = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        _unsafe_view_14 = torch.ops.aten._unsafe_view.default(clone_18, [1, 512, 384]);  clone_18 = None
        permute_142 = torch.ops.aten.permute.default(permute_140, [3, 4, 0, 2, 1]);  permute_140 = None
        clone_19 = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        _unsafe_view_15 = torch.ops.aten._unsafe_view.default(clone_19, [1, 384, 384]);  clone_19 = None
        bmm_15 = torch.ops.aten.bmm.default(_unsafe_view_14, _unsafe_view_15);  _unsafe_view_14 = _unsafe_view_15 = None
        view_260 = torch.ops.aten.view.default(bmm_15, [512, 1, 1, 1, 384]);  bmm_15 = None
        permute_143 = torch.ops.aten.permute.default(view_260, [3, 0, 4, 1, 2]);  view_260 = None
        view_261 = torch.ops.aten.view.default(permute_143, [1, 512, 384]);  permute_143 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        mul_32 = torch.ops.aten.mul.Tensor(view_261, unsqueeze_76);  view_261 = unsqueeze_76 = None
        add_37 = torch.ops.aten.add.Tensor(add_29, mul_32);  mul_32 = None
        split_tensor_51 = torch.ops.aten.split.Tensor(add_29, 512, dim = -2);  add_29 = None
        getitem_320 = split_tensor_51[0];  split_tensor_51 = None
        _to_copy_141 = torch.ops.aten._to_copy.default(getitem_320, dtype = torch.float32);  getitem_320 = None
        native_layer_norm_default_31 = torch.ops.aten.native_layer_norm.default(_to_copy_141, [384], arg90_1, arg91_1, 1e-05);  _to_copy_141 = arg90_1 = arg91_1 = None
        getitem_321 = native_layer_norm_default_31[0]
        _to_copy_142 = torch.ops.aten._to_copy.default(arg92_1, dtype = torch.bfloat16);  arg92_1 = None
        _to_copy_143 = torch.ops.aten._to_copy.default(getitem_321, dtype = torch.bfloat16);  getitem_321 = None
        t_52 = torch.ops.aten.t.default(_to_copy_142);  _to_copy_142 = None
        view_262 = torch.ops.aten.view.default(_to_copy_143, [512, 384]);  _to_copy_143 = None
        mm_52 = torch.ops.aten.mm.default(view_262, t_52);  view_262 = t_52 = None
        view_263 = torch.ops.aten.view.default(mm_52, [1, 512, 1536]);  mm_52 = None
        split_tensor_52 = torch.ops.aten.split.Tensor(view_263, 768, dim = -1);  view_263 = None
        getitem_324 = split_tensor_52[0]
        getitem_325 = split_tensor_52[1];  split_tensor_52 = None
        silu_7 = torch.ops.aten.silu.default(getitem_324);  getitem_324 = None
        mul_33 = torch.ops.aten.mul.Tensor(silu_7, getitem_325);  silu_7 = getitem_325 = None
        _to_copy_144 = torch.ops.aten._to_copy.default(arg93_1, dtype = torch.bfloat16);  arg93_1 = None
        t_53 = torch.ops.aten.t.default(_to_copy_144);  _to_copy_144 = None
        view_265 = torch.ops.aten.view.default(mul_33, [512, 768]);  mul_33 = None
        mm_53 = torch.ops.aten.mm.default(view_265, t_53);  view_265 = t_53 = None
        view_266 = torch.ops.aten.view.default(mm_53, [1, 512, 384]);  mm_53 = None
        add_38 = torch.ops.aten.add.Tensor(add_37, view_266);  add_37 = view_266 = None
        _to_copy_145 = torch.ops.aten._to_copy.default(add_38, dtype = torch.float32);  add_38 = None
        native_layer_norm_default_32 = torch.ops.aten.native_layer_norm.default(_to_copy_145, [384], None, None, 1e-05);  _to_copy_145 = None
        getitem_326 = native_layer_norm_default_32[0]
        _to_copy_146 = torch.ops.aten._to_copy.default(add_34, dtype = torch.float32);  add_34 = None
        native_layer_norm_default_33 = torch.ops.aten.native_layer_norm.default(_to_copy_146, [256], None, None, 1e-05);  _to_copy_146 = None
        getitem_329 = native_layer_norm_default_33[0]
        _to_copy_147 = torch.ops.aten._to_copy.default(arg103_1, dtype = torch.bfloat16);  arg103_1 = None
        _to_copy_148 = torch.ops.aten._to_copy.default(getitem_329, dtype = torch.bfloat16)
        t_54 = torch.ops.aten.t.default(_to_copy_147);  _to_copy_147 = None
        view_267 = torch.ops.aten.view.default(_to_copy_148, [262144, 256]);  _to_copy_148 = None
        mm_54 = torch.ops.aten.mm.default(view_267, t_54);  view_267 = t_54 = None
        view_268 = torch.ops.aten.view.default(mm_54, [1, 512, 512, 64]);  mm_54 = None
        permute_144 = torch.ops.aten.permute.default(getitem_329, [0, 2, 1, 3])
        add_39 = torch.ops.aten.add.Tensor(getitem_329, permute_144);  getitem_329 = permute_144 = None
        _to_copy_149 = torch.ops.aten._to_copy.default(arg104_1, dtype = torch.bfloat16);  arg104_1 = None
        _to_copy_150 = torch.ops.aten._to_copy.default(add_39, dtype = torch.bfloat16);  add_39 = None
        t_55 = torch.ops.aten.t.default(_to_copy_149);  _to_copy_149 = None
        view_269 = torch.ops.aten.view.default(_to_copy_150, [262144, 256]);  _to_copy_150 = None
        mm_55 = torch.ops.aten.mm.default(view_269, t_55);  view_269 = t_55 = None
        view_270 = torch.ops.aten.view.default(mm_55, [1, 512, 512, 64]);  mm_55 = None
        _to_copy_151 = torch.ops.aten._to_copy.default(arg102_1, dtype = torch.bfloat16);  arg102_1 = None
        _to_copy_152 = torch.ops.aten._to_copy.default(getitem_326, dtype = torch.bfloat16);  getitem_326 = None
        t_56 = torch.ops.aten.t.default(_to_copy_151);  _to_copy_151 = None
        view_271 = torch.ops.aten.view.default(_to_copy_152, [512, 384]);  _to_copy_152 = None
        mm_56 = torch.ops.aten.mm.default(view_271, t_56);  view_271 = t_56 = None
        view_272 = torch.ops.aten.view.default(mm_56, [1, 512, 1850]);  mm_56 = None
        view_273 = torch.ops.aten.view.default(view_272, [1, 512, 37, 50]);  view_272 = None
        arange_2 = torch.ops.aten.arange.default(1, device = self.device, pin_memory = False)
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(arange_2, 1);  arange_2 = None
        index_2 = torch.ops.aten.index.Tensor(view_273, [unsqueeze_77, arg113_1, arg114_1]);  view_273 = unsqueeze_77 = arg113_1 = arg114_1 = None
        return (view_268, view_270, index_2)
        
    # To see more debug info, please use `graph_module.print_readable()`