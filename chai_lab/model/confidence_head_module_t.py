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

        _to_copy = arg0_1.to(dtype = torch.bfloat16) ;  arg0_1 = None
        t = _to_copy.t() ;  _to_copy = None
        view = arg106_1.view(512, 384) ;  arg106_1 = None
        mm = torch.mm(view,t) ;  view = t = None
        view_1 = mm.view(1, 512, 512) ;  mm = None
        split_tensor = torch.split(view_1,256,dim = -1) ;  view_1 = None
        getitem = split_tensor[0]
        getitem_1 = split_tensor[1];  split_tensor = None
        view_2 = getitem.view(1, 512, 1, 256) ;  getitem = None
        view_3 = getitem_1.view(1, 1, 512, 256) ;  getitem_1 = None
        add = torch.add(arg108_1,view_2) ;  arg108_1 = view_2 = None
        add_1 = torch.add(add,view_3) ;  add = view_3 = None
        arange = torch.arange(1,device = self.device,pin_memory = False) 
        view_4 = arange.view(1, 1) ;  arange = None
        index = arg111_1[view_4,arg112_1] ;  arg111_1 = view_4 = arg112_1 = None
        _cdist_forward = torch.cdist(index, index, p = 2.0) ;  index = None
        searchsorted = torch.searchsorted(arg105_1,_cdist_forward) ;  arg105_1 = _cdist_forward = None
        arange_1 = torch.arange(16,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze = torch.unsqueeze(searchsorted,-1) ;  searchsorted = None
        eq = (unsqueeze == arange_1) ;  unsqueeze = arange_1 = None
        _to_copy_1 = eq.to(dtype = torch.int64) ;  eq = None
        _to_copy_2 = _to_copy_1.to(dtype = torch.float32) ;  _to_copy_1 = None
        _to_copy_3 = _to_copy_2.to(dtype = torch.bfloat16) ;  _to_copy_2 = None
        _to_copy_4 = arg1_1.to(dtype = torch.bfloat16) ;  arg1_1 = None
        t_1 = _to_copy_4.t() ;  _to_copy_4 = None
        view_5 = _to_copy_3.view(262144, 16) ;  _to_copy_3 = None
        mm_1 = torch.mm(view_5,t_1) ;  view_5 = t_1 = None
        view_6 = mm_1.view(1, 512, 512, 256) ;  mm_1 = None
        add_2 = torch.add(add_1,view_6) ;  add_1 = view_6 = None
        view_7 = arg109_1.view(1, 512, 1) 
        view_8 = arg109_1.view(1, 1, 512) 
        bitwise_and_1 = torch.bitwise_and(view_7,view_8) ;  view_7 = view_8 = None
        _to_copy_5 = add_2.to(dtype = torch.float32) 
        native_layer_norm_default = (torch.nn.functional.layer_norm(_to_copy_5,[256],arg6_1,arg7_1,1e-05),) ;  _to_copy_5 = arg6_1 = arg7_1 = None
        getitem_2 = native_layer_norm_default[0]
        split_with_sizes_default = torch.split(arg9_1,[512,512]) ;  arg9_1 = None
        getitem_5 = split_with_sizes_default[0]
        getitem_6 = split_with_sizes_default[1];  split_with_sizes_default = None
        split_with_sizes_default_1 = torch.split(arg10_1,[512,512,256]) ;  arg10_1 = None
        getitem_7 = split_with_sizes_default_1[0]
        getitem_8 = split_with_sizes_default_1[1]
        getitem_9 = split_with_sizes_default_1[2];  split_with_sizes_default_1 = None
        _to_copy_6 = getitem_5.to(dtype = torch.bfloat16) ;  getitem_5 = None
        _to_copy_7 = getitem_2.to(dtype = torch.bfloat16) 
        t_2 = _to_copy_6.t() ;  _to_copy_6 = None
        view_9 = _to_copy_7.view(262144, 256) ;  _to_copy_7 = None
        mm_2 = torch.mm(view_9,t_2) ;  view_9 = t_2 = None
        view_10 = mm_2.view(1, 512, 512, 512) ;  mm_2 = None
        _to_copy_8 = getitem_7.to(dtype = torch.bfloat16) ;  getitem_7 = None
        _to_copy_9 = getitem_2.to(dtype = torch.bfloat16) 
        t_3 = _to_copy_8.t() ;  _to_copy_8 = None
        view_11 = _to_copy_9.view(262144, 256) ;  _to_copy_9 = None
        mm_3 = torch.mm(view_11,t_3) ;  view_11 = t_3 = None
        view_12 = mm_3.view(1, 512, 512, 512) ;  mm_3 = None
        sigmoid = torch.sigmoid(view_12) ;  view_12 = None
        mul_2 = torch.mul(view_10,sigmoid) ;  view_10 = sigmoid = None
        unsqueeze_1 = torch.unsqueeze(bitwise_and_1,3) 
        bitwise_not = torch.bitwise_not(unsqueeze_1) ;  unsqueeze_1 = None
        masked_fill = mul_2.masked_fill(bitwise_not,0) ;  mul_2 = bitwise_not = None
        split_tensor_1 = torch.split(masked_fill,256,dim = -1) 
        getitem_12 = split_tensor_1[0]
        unsqueeze_4 = torch.unsqueeze(getitem_12,4) ;  getitem_12 = None
        permute_4 = unsqueeze_4.permute(0, 1, 4, 3, 2) ;  unsqueeze_4 = None
        permute_5 = permute_4.permute(3, 1, 4, 0, 2) ;  permute_4 = None
        view_15 = permute_5.view(256, 512, 512) ;  permute_5 = None
        split_tensor_2 = torch.split(masked_fill,256,dim = -1) ;  masked_fill = None
        getitem_15 = split_tensor_2[1];  split_tensor_2 = None
        unsqueeze_5 = torch.unsqueeze(getitem_15,4) ;  getitem_15 = None
        permute_6 = unsqueeze_5.permute(0, 4, 1, 3, 2) ;  unsqueeze_5 = None
        permute_7 = permute_6.permute(3, 4, 0, 2, 1) ;  permute_6 = None
        view_16 = permute_7.view(256, 512, 512) ;  permute_7 = None
        bmm = torch.bmm(view_15,view_16) ;  view_15 = view_16 = None
        view_17 = bmm.view(256, 512, 1, 1, 512) ;  bmm = None
        permute_8 = view_17.permute(3, 1, 4, 0, 2) ;  view_17 = None
        view_18 = permute_8.view(1, 512, 512, 256) ;  permute_8 = None
        _to_copy_10 = getitem_6.to(dtype = torch.bfloat16) ;  getitem_6 = None
        _to_copy_11 = getitem_2.to(dtype = torch.bfloat16) 
        t_4 = _to_copy_10.t() ;  _to_copy_10 = None
        view_19 = _to_copy_11.view(262144, 256) ;  _to_copy_11 = None
        mm_4 = torch.mm(view_19,t_4) ;  view_19 = t_4 = None
        view_20 = mm_4.view(1, 512, 512, 512) ;  mm_4 = None
        _to_copy_12 = getitem_8.to(dtype = torch.bfloat16) ;  getitem_8 = None
        _to_copy_13 = getitem_2.to(dtype = torch.bfloat16) 
        t_5 = _to_copy_12.t() ;  _to_copy_12 = None
        view_21 = _to_copy_13.view(262144, 256) ;  _to_copy_13 = None
        mm_5 = torch.mm(view_21,t_5) ;  view_21 = t_5 = None
        view_22 = mm_5.view(1, 512, 512, 512) ;  mm_5 = None
        sigmoid_1 = torch.sigmoid(view_22) ;  view_22 = None
        mul_3 = torch.mul(view_20,sigmoid_1) ;  view_20 = sigmoid_1 = None
        view_23 = mul_3.view(262144, 512) ;  mul_3 = None
        view_24 = view_23.view(1, 512, 512, 512) ;  view_23 = None
        transpose = torch.transpose(bitwise_and_1,1,2) 
        unsqueeze_6 = torch.unsqueeze(transpose,3) ;  transpose = None
        clone = torch.clone(unsqueeze_6,memory_format = torch.contiguous_format) ;  unsqueeze_6 = None
        bitwise_not_1 = torch.bitwise_not(clone) ;  clone = None
        masked_fill_1 = view_24.masked_fill(bitwise_not_1,0) ;  view_24 = bitwise_not_1 = None
        view_25 = masked_fill_1.view(262144, 512) ;  masked_fill_1 = None
        view_29 = view_25.view(1, 512, 512, 512) 
        split_tensor_3 = torch.split(view_29,256,dim = -1) ;  view_29 = None
        getitem_18 = split_tensor_3[0]
        unsqueeze_9 = torch.unsqueeze(getitem_18,4) ;  getitem_18 = None
        permute_13 = unsqueeze_9.permute(0, 2, 4, 3, 1) ;  unsqueeze_9 = None
        permute_14 = permute_13.permute(3, 1, 4, 0, 2) ;  permute_13 = None
        view_30 = permute_14.view(256, 512, 512) ;  permute_14 = None
        view_31 = view_25.view(1, 512, 512, 512) ;  view_25 = None
        split_tensor_4 = torch.split(view_31,256,dim = -1) ;  view_31 = None
        getitem_21 = split_tensor_4[1];  split_tensor_4 = None
        unsqueeze_10 = torch.unsqueeze(getitem_21,4) ;  getitem_21 = None
        permute_15 = unsqueeze_10.permute(0, 4, 2, 3, 1) ;  unsqueeze_10 = None
        permute_16 = permute_15.permute(3, 4, 0, 2, 1) ;  permute_15 = None
        view_32 = permute_16.view(256, 512, 512) ;  permute_16 = None
        bmm_1 = torch.bmm(view_30,view_32) ;  view_30 = view_32 = None
        view_33 = bmm_1.view(256, 512, 1, 1, 512) ;  bmm_1 = None
        permute_17 = view_33.permute(3, 1, 4, 0, 2) ;  view_33 = None
        view_34 = permute_17.view(1, 512, 512, 256) ;  permute_17 = None
        _to_copy_14 = view_18.to(dtype = torch.float32) ;  view_18 = None
        native_layer_norm_default_1 = (torch.nn.functional.layer_norm(_to_copy_14,[256],None,None,1e-05),) ;  _to_copy_14 = None
        getitem_22 = native_layer_norm_default_1[0]
        _to_copy_15 = view_34.to(dtype = torch.float32) ;  view_34 = None
        native_layer_norm_default_2 = (torch.nn.functional.layer_norm(_to_copy_15,[256],None,None,1e-05),) ;  _to_copy_15 = None
        getitem_25 = native_layer_norm_default_2[0]
        add_3 = torch.add(getitem_22,getitem_25) ;  getitem_22 = getitem_25 = None
        _to_copy_16 = arg8_1.to(dtype = torch.bfloat16) ;  arg8_1 = None
        _to_copy_17 = add_3.to(dtype = torch.bfloat16) ;  add_3 = None
        t_6 = _to_copy_16.t() ;  _to_copy_16 = None
        view_35 = _to_copy_17.view(262144, 256) ;  _to_copy_17 = None
        mm_6 = torch.mm(view_35,t_6) ;  view_35 = t_6 = None
        view_36 = mm_6.view(1, 512, 512, 256) ;  mm_6 = None
        _to_copy_18 = getitem_9.to(dtype = torch.bfloat16) ;  getitem_9 = None
        _to_copy_19 = getitem_2.to(dtype = torch.bfloat16) ;  getitem_2 = None
        t_7 = _to_copy_18.t() ;  _to_copy_18 = None
        view_37 = _to_copy_19.view(262144, 256) ;  _to_copy_19 = None
        mm_7 = torch.mm(view_37,t_7) ;  view_37 = t_7 = None
        view_38 = mm_7.view(1, 512, 512, 256) ;  mm_7 = None
        sigmoid_2 = torch.sigmoid(view_38) ;  view_38 = None
        mul_4 = torch.mul(view_36,sigmoid_2) ;  view_36 = sigmoid_2 = None
        add_4 = torch.add(add_2,mul_4) ;  mul_4 = None
        _to_copy_20 = add_2.to(dtype = torch.float32) 
        native_layer_norm_default_3 = (torch.nn.functional.layer_norm(_to_copy_20,[256],arg11_1,arg12_1,1e-05),) ;  _to_copy_20 = arg11_1 = arg12_1 = None
        getitem_28 = native_layer_norm_default_3[0]
        _to_copy_21 = arg13_1.to(dtype = torch.bfloat16) ;  arg13_1 = None
        _to_copy_22 = getitem_28.to(dtype = torch.bfloat16) ;  getitem_28 = None
        t_8 = _to_copy_21.t() ;  _to_copy_21 = None
        view_39 = _to_copy_22.view(262144, 256) ;  _to_copy_22 = None
        mm_8 = torch.mm(view_39,t_8) ;  view_39 = t_8 = None
        view_40 = mm_8.view(1, 512, 512, 2056) ;  mm_8 = None
        split_with_sizes_default_2 = torch.split(view_40,[2048,8],dim = -1) ;  view_40 = None
        getitem_31 = split_with_sizes_default_2[0]
        getitem_32 = split_with_sizes_default_2[1];  split_with_sizes_default_2 = None
        view_41 = getitem_32.view(1, 512, 512, 2, 4) ;  getitem_32 = None
        permute_18 = view_41.permute(0, 3, 4, 1, 2) ;  view_41 = None
        view_42 = permute_18.view(1, 2, 4, 1, 512, 512) ;  permute_18 = None
        view_43 = bitwise_and_1.view(1, 1, 1, 1, 512, 512) 
        bitwise_not_2 = torch.bitwise_not(view_43) ;  view_43 = None
        masked_fill_2 = view_42.masked_fill(bitwise_not_2,-10000) ;  view_42 = bitwise_not_2 = None
        view_44 = masked_fill_2.view(1, 2, 4, 512, 512) ;  masked_fill_2 = None
        view_45 = view_44.view(8, 1, 512, 512) ;  view_44 = None
        split_tensor_5 = torch.split(getitem_31,1024,dim = -1) ;  getitem_31 = None
        getitem_33 = split_tensor_5[0]
        getitem_34 = split_tensor_5[1];  split_tensor_5 = None
        permute_19 = getitem_34.permute(0, 2, 1, 3) ;  getitem_34 = None
        stack = torch.stack([getitem_33,permute_19]) ;  getitem_33 = permute_19 = None
        view_46 = stack.view(2, 1, 512, 512, 4, 4, 64) ;  stack = None
        permute_20 = view_46.permute(4, 1, 0, 5, 2, 3, 6) ;  view_46 = None
        clone_1 = torch.clone(permute_20,memory_format = torch.contiguous_format) ;  permute_20 = None
        _unsafe_view = clone_1.view(4, 8, 512, 512, 64) ;  clone_1 = None
        unbind_int = torch.unbind(_unsafe_view) ;  _unsafe_view = None
        getitem_35 = unbind_int[0]
        getitem_36 = unbind_int[1]
        getitem_37 = unbind_int[2]
        getitem_38 = unbind_int[3];  unbind_int = None
        split_tensor_6 = torch.split(getitem_35,4) ;  getitem_35 = None
        getitem_39 = split_tensor_6[0]
        getitem_40 = split_tensor_6[1];  split_tensor_6 = None
        split_tensor_7 = torch.split(getitem_36,4) ;  getitem_36 = None
        getitem_41 = split_tensor_7[0]
        getitem_42 = split_tensor_7[1];  split_tensor_7 = None
        split_tensor_8 = torch.split(getitem_37,4) ;  getitem_37 = None
        getitem_43 = split_tensor_8[0]
        getitem_44 = split_tensor_8[1];  split_tensor_8 = None
        split_tensor_9 = torch.split(view_45,4) ;  view_45 = None
        getitem_45 = split_tensor_9[0]
        getitem_46 = split_tensor_9[1];  split_tensor_9 = None
        expand = getitem_45.expand(4, 512, 512, 512) ;  getitem_45 = None
        _scaled_dot_product_efficient_attention_default = (torch.nn.functional.scaled_dot_product_attention(getitem_39,getitem_41,getitem_43,expand,False),) ;  getitem_39 = getitem_41 = getitem_43 = expand = None
        getitem_47 = _scaled_dot_product_efficient_attention_default[0]
        expand_1 = getitem_46.expand(4, 512, 512, 512) ;  getitem_46 = None
        _scaled_dot_product_efficient_attention_default_1 = (torch.nn.functional.scaled_dot_product_attention(getitem_40,getitem_42,getitem_44,expand_1,False),) ;  getitem_40 = getitem_42 = getitem_44 = expand_1 = None
        getitem_51 = _scaled_dot_product_efficient_attention_default_1[0]
        cat = torch.cat([getitem_47,getitem_51]) ;  getitem_47 = getitem_51 = None
        sigmoid_3 = torch.sigmoid(getitem_38) ;  getitem_38 = None
        mul_5 = torch.mul(cat,sigmoid_3) ;  cat = sigmoid_3 = None
        view_47 = mul_5.view(1, 2, 4, 512, 512, 64) ;  mul_5 = None
        permute_21 = view_47.permute(0, 3, 4, 1, 2, 5) ;  view_47 = None
        clone_2 = torch.clone(permute_21,memory_format = torch.contiguous_format) ;  permute_21 = None
        _unsafe_view_1 = clone_2.view(1, 512, 512, 512) ;  clone_2 = None
        _to_copy_23 = arg14_1.to(dtype = torch.bfloat16) ;  arg14_1 = None
        t_9 = _to_copy_23.t() ;  _to_copy_23 = None
        view_48 = _unsafe_view_1.view(262144, 512) ;  _unsafe_view_1 = None
        mm_9 = torch.mm(view_48,t_9) ;  view_48 = t_9 = None
        view_49 = mm_9.view(1, 512, 512, 512) ;  mm_9 = None
        view_50 = view_49.view(1, 512, 512, 2, 4, 64) ;  view_49 = None
        permute_22 = view_50.permute(3, 0, 1, 2, 4, 5) ;  view_50 = None
        view_51 = permute_22.view(2, 1, 512, 512, 256) ;  permute_22 = None
        unbind_int_1 = torch.unbind(view_51) ;  view_51 = None
        getitem_55 = unbind_int_1[0]
        getitem_56 = unbind_int_1[1];  unbind_int_1 = None
        permute_23 = getitem_56.permute(0, 2, 1, 3) ;  getitem_56 = None
        permute_24 = permute_23.permute(0, 2, 1, 3) ;  permute_23 = None
        add_5 = torch.add(getitem_55,permute_24) ;  getitem_55 = permute_24 = None
        add_6 = torch.add(add_4,add_5) ;  add_4 = add_5 = None
        split_tensor_10 = torch.split(add_2,512,dim = -2) 
        getitem_57 = split_tensor_10[0];  split_tensor_10 = None
        _to_copy_24 = getitem_57.to(dtype = torch.float32) ;  getitem_57 = None
        native_layer_norm_default_4 = (torch.nn.functional.layer_norm(_to_copy_24,[256],arg2_1,arg3_1,1e-05),) ;  _to_copy_24 = arg2_1 = arg3_1 = None
        getitem_58 = native_layer_norm_default_4[0]
        _to_copy_25 = arg4_1.to(dtype = torch.bfloat16) ;  arg4_1 = None
        _to_copy_26 = getitem_58.to(dtype = torch.bfloat16) ;  getitem_58 = None
        t_10 = _to_copy_25.t() ;  _to_copy_25 = None
        view_52 = _to_copy_26.view(262144, 256) ;  _to_copy_26 = None
        mm_10 = torch.mm(view_52,t_10) ;  view_52 = t_10 = None
        view_53 = mm_10.view(1, 512, 512, 1024) ;  mm_10 = None
        split_tensor_11 = torch.split(view_53,512,dim = -1) ;  view_53 = None
        getitem_61 = split_tensor_11[0]
        getitem_62 = split_tensor_11[1];  split_tensor_11 = None
        silu = torch.nn.functional.silu(getitem_61) ;  getitem_61 = None
        mul_6 = torch.mul(silu,getitem_62) ;  silu = getitem_62 = None
        _to_copy_27 = arg5_1.to(dtype = torch.bfloat16) ;  arg5_1 = None
        t_11 = _to_copy_27.t() ;  _to_copy_27 = None
        view_55 = mul_6.view(262144, 512) ;  mul_6 = None
        mm_11 = torch.mm(view_55,t_11) ;  view_55 = t_11 = None
        view_56 = mm_11.view(1, 512, 512, 256) ;  mm_11 = None
        add_7 = torch.add(add_6,view_56) ;  add_6 = view_56 = None
        _to_copy_28 = arg107_1.to(dtype = torch.float32) 
        native_layer_norm_default_5 = (torch.nn.functional.layer_norm(_to_copy_28,[384],arg19_1,arg20_1,1e-05),) ;  _to_copy_28 = arg19_1 = arg20_1 = None
        getitem_63 = native_layer_norm_default_5[0]
        _to_copy_29 = add_2.to(dtype = torch.float32) ;  add_2 = None
        native_layer_norm_default_6 = (torch.nn.functional.layer_norm(_to_copy_29,[256],arg21_1,arg22_1,1e-05),) ;  _to_copy_29 = arg21_1 = arg22_1 = None
        getitem_66 = native_layer_norm_default_6[0]
        _to_copy_30 = arg23_1.to(dtype = torch.bfloat16) ;  arg23_1 = None
        _to_copy_31 = getitem_66.to(dtype = torch.bfloat16) ;  getitem_66 = None
        t_12 = _to_copy_30.t() ;  _to_copy_30 = None
        view_57 = _to_copy_31.view(262144, 256) ;  _to_copy_31 = None
        mm_12 = torch.mm(view_57,t_12) ;  view_57 = t_12 = None
        view_58 = mm_12.view(1, 512, 512, 16) ;  mm_12 = None
        permute_25 = view_58.permute(0, 3, 1, 2) ;  view_58 = None
        view_59 = bitwise_and_1.view(1, 1, 512, 512) ;  bitwise_and_1 = None
        bitwise_not_3 = torch.bitwise_not(view_59) ;  view_59 = None
        masked_fill_3 = permute_25.masked_fill(bitwise_not_3,-10000) ;  permute_25 = bitwise_not_3 = None
        _to_copy_32 = getitem_63.to(dtype = torch.bfloat16) ;  getitem_63 = None
        _to_copy_33 = arg25_1.to(dtype = torch.bfloat16) ;  arg25_1 = None
        unsqueeze_11 = torch.unsqueeze(_to_copy_32,3) ;  _to_copy_32 = None
        unsqueeze_12 = torch.unsqueeze(unsqueeze_11,4) ;  unsqueeze_11 = None
        unsqueeze_13 = torch.unsqueeze(unsqueeze_12,5) ;  unsqueeze_12 = None
        permute_26 = unsqueeze_13.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_13 = None
        unsqueeze_14 = torch.unsqueeze(_to_copy_33,4) ;  _to_copy_33 = None
        unsqueeze_15 = torch.unsqueeze(unsqueeze_14,5) ;  unsqueeze_14 = None
        permute_27 = unsqueeze_15.permute(1, 4, 2, 5, 3, 0) ;  unsqueeze_15 = None
        permute_28 = permute_26.permute(3, 5, 0, 1, 2, 4) ;  permute_26 = None
        view_60 = permute_28.view(1, 512, 384) ;  permute_28 = None
        permute_29 = permute_27.permute(5, 0, 1, 2, 4, 3) ;  permute_27 = None
        view_61 = permute_29.view(1, 384, 1536) ;  permute_29 = None
        bmm_2 = torch.bmm(view_60,view_61) ;  view_60 = view_61 = None
        view_62 = bmm_2.view(512, 1, 4, 1, 16, 24) ;  bmm_2 = None
        permute_30 = view_62.permute(2, 3, 4, 0, 5, 1) ;  view_62 = None
        view_63 = permute_30.view(4, 1, 16, 512, 24) ;  permute_30 = None
        unbind_int_2 = torch.unbind(view_63) ;  view_63 = None
        getitem_69 = unbind_int_2[0]
        getitem_70 = unbind_int_2[1]
        getitem_71 = unbind_int_2[2]
        getitem_72 = unbind_int_2[3];  unbind_int_2 = None
        view_64 = arg24_1.view(1, 16, 1, 24) ;  arg24_1 = None
        add_8 = torch.add(getitem_69,view_64) ;  getitem_69 = view_64 = None
        _to_copy_34 = add_8.to(dtype = torch.bfloat16) ;  add_8 = None
        expand_2 = masked_fill_3.expand(1, 16, 512, 512) ;  masked_fill_3 = None
        _scaled_dot_product_efficient_attention_default_2 = (torch.nn.functional.scaled_dot_product_attention(_to_copy_34,getitem_70,getitem_71,expand_2,False),) ;  _to_copy_34 = getitem_70 = getitem_71 = expand_2 = None
        getitem_73 = _scaled_dot_product_efficient_attention_default_2[0]
        add_9 = torch.add(getitem_72,1) ;  getitem_72 = None
        sigmoid_4 = torch.sigmoid(add_9) ;  add_9 = None
        mul_7 = torch.mul(getitem_73,sigmoid_4) ;  getitem_73 = sigmoid_4 = None
        _to_copy_35 = arg26_1.to(dtype = torch.bfloat16) ;  arg26_1 = None
        unsqueeze_16 = torch.unsqueeze(mul_7,4) ;  mul_7 = None
        permute_31 = unsqueeze_16.permute(0, 2, 4, 3, 1) ;  unsqueeze_16 = None
        unsqueeze_17 = torch.unsqueeze(_to_copy_35,3) ;  _to_copy_35 = None
        unsqueeze_18 = torch.unsqueeze(unsqueeze_17,4) ;  unsqueeze_17 = None
        permute_32 = unsqueeze_18.permute(3, 4, 2, 1, 0) ;  unsqueeze_18 = None
        permute_33 = permute_31.permute(1, 3, 4, 0, 2) ;  permute_31 = None
        clone_3 = torch.clone(permute_33,memory_format = torch.contiguous_format) ;  permute_33 = None
        _unsafe_view_2 = clone_3.view(1, 512, 384) ;  clone_3 = None
        permute_34 = permute_32.permute(3, 4, 0, 2, 1) ;  permute_32 = None
        clone_4 = torch.clone(permute_34,memory_format = torch.contiguous_format) ;  permute_34 = None
        _unsafe_view_3 = clone_4.view(1, 384, 384) ;  clone_4 = None
        bmm_3 = torch.bmm(_unsafe_view_2,_unsafe_view_3) ;  _unsafe_view_2 = _unsafe_view_3 = None
        view_65 = bmm_3.view(512, 1, 1, 1, 384) ;  bmm_3 = None
        permute_35 = view_65.permute(3, 0, 4, 1, 2) ;  view_65 = None
        view_66 = permute_35.view(1, 512, 384) ;  permute_35 = None
        unsqueeze_19 = torch.unsqueeze(arg109_1,-1) 
        mul_8 = torch.mul(view_66,unsqueeze_19) ;  view_66 = unsqueeze_19 = None
        add_10 = torch.add(arg107_1,mul_8) ;  mul_8 = None
        split_tensor_12 = torch.split(arg107_1,512,dim = -2) ;  arg107_1 = None
        getitem_77 = split_tensor_12[0];  split_tensor_12 = None
        _to_copy_36 = getitem_77.to(dtype = torch.float32) ;  getitem_77 = None
        native_layer_norm_default_7 = (torch.nn.functional.layer_norm(_to_copy_36,[384],arg15_1,arg16_1,1e-05),) ;  _to_copy_36 = arg15_1 = arg16_1 = None
        getitem_78 = native_layer_norm_default_7[0]
        _to_copy_37 = arg17_1.to(dtype = torch.bfloat16) ;  arg17_1 = None
        _to_copy_38 = getitem_78.to(dtype = torch.bfloat16) ;  getitem_78 = None
        t_13 = _to_copy_37.t() ;  _to_copy_37 = None
        view_67 = _to_copy_38.view(512, 384) ;  _to_copy_38 = None
        mm_13 = torch.mm(view_67,t_13) ;  view_67 = t_13 = None
        view_68 = mm_13.view(1, 512, 1536) ;  mm_13 = None
        split_tensor_13 = torch.split(view_68,768,dim = -1) ;  view_68 = None
        getitem_81 = split_tensor_13[0]
        getitem_82 = split_tensor_13[1];  split_tensor_13 = None
        silu_1 = torch.nn.functional.silu(getitem_81) ;  getitem_81 = None
        mul_9 = torch.mul(silu_1,getitem_82) ;  silu_1 = getitem_82 = None
        _to_copy_39 = arg18_1.to(dtype = torch.bfloat16) ;  arg18_1 = None
        t_14 = _to_copy_39.t() ;  _to_copy_39 = None
        view_70 = mul_9.view(512, 768) ;  mul_9 = None
        mm_14 = torch.mm(view_70,t_14) ;  view_70 = t_14 = None
        view_71 = mm_14.view(1, 512, 384) ;  mm_14 = None
        add_11 = torch.add(add_10,view_71) ;  add_10 = view_71 = None
        view_72 = arg109_1.view(1, 512, 1) 
        view_73 = arg109_1.view(1, 1, 512) 
        bitwise_and_2 = torch.bitwise_and(view_72,view_73) ;  view_72 = view_73 = None
        _to_copy_40 = add_7.to(dtype = torch.float32) 
        native_layer_norm_default_8 = (torch.nn.functional.layer_norm(_to_copy_40,[256],arg31_1,arg32_1,1e-05),) ;  _to_copy_40 = arg31_1 = arg32_1 = None
        getitem_83 = native_layer_norm_default_8[0]
        split_with_sizes_default_3 = torch.split(arg34_1,[512,512]) ;  arg34_1 = None
        getitem_86 = split_with_sizes_default_3[0]
        getitem_87 = split_with_sizes_default_3[1];  split_with_sizes_default_3 = None
        split_with_sizes_default_4 = torch.split(arg35_1,[512,512,256]) ;  arg35_1 = None
        getitem_88 = split_with_sizes_default_4[0]
        getitem_89 = split_with_sizes_default_4[1]
        getitem_90 = split_with_sizes_default_4[2];  split_with_sizes_default_4 = None
        _to_copy_41 = getitem_86.to(dtype = torch.bfloat16) ;  getitem_86 = None
        _to_copy_42 = getitem_83.to(dtype = torch.bfloat16) 
        t_15 = _to_copy_41.t() ;  _to_copy_41 = None
        view_74 = _to_copy_42.view(262144, 256) ;  _to_copy_42 = None
        mm_15 = torch.mm(view_74,t_15) ;  view_74 = t_15 = None
        view_75 = mm_15.view(1, 512, 512, 512) ;  mm_15 = None
        _to_copy_43 = getitem_88.to(dtype = torch.bfloat16) ;  getitem_88 = None
        _to_copy_44 = getitem_83.to(dtype = torch.bfloat16) 
        t_16 = _to_copy_43.t() ;  _to_copy_43 = None
        view_76 = _to_copy_44.view(262144, 256) ;  _to_copy_44 = None
        mm_16 = torch.mm(view_76,t_16) ;  view_76 = t_16 = None
        view_77 = mm_16.view(1, 512, 512, 512) ;  mm_16 = None
        sigmoid_5 = torch.sigmoid(view_77) ;  view_77 = None
        mul_10 = torch.mul(view_75,sigmoid_5) ;  view_75 = sigmoid_5 = None
        unsqueeze_20 = torch.unsqueeze(bitwise_and_2,3) 
        bitwise_not_4 = torch.bitwise_not(unsqueeze_20) ;  unsqueeze_20 = None
        masked_fill_4 = mul_10.masked_fill(bitwise_not_4,0) ;  mul_10 = bitwise_not_4 = None
        split_tensor_14 = torch.split(masked_fill_4,256,dim = -1) 
        getitem_93 = split_tensor_14[0]
        unsqueeze_23 = torch.unsqueeze(getitem_93,4) ;  getitem_93 = None
        permute_40 = unsqueeze_23.permute(0, 1, 4, 3, 2) ;  unsqueeze_23 = None
        permute_41 = permute_40.permute(3, 1, 4, 0, 2) ;  permute_40 = None
        view_80 = permute_41.view(256, 512, 512) ;  permute_41 = None
        split_tensor_15 = torch.split(masked_fill_4,256,dim = -1) ;  masked_fill_4 = None
        getitem_96 = split_tensor_15[1];  split_tensor_15 = None
        unsqueeze_24 = torch.unsqueeze(getitem_96,4) ;  getitem_96 = None
        permute_42 = unsqueeze_24.permute(0, 4, 1, 3, 2) ;  unsqueeze_24 = None
        permute_43 = permute_42.permute(3, 4, 0, 2, 1) ;  permute_42 = None
        view_81 = permute_43.view(256, 512, 512) ;  permute_43 = None
        bmm_4 = torch.bmm(view_80,view_81) ;  view_80 = view_81 = None
        view_82 = bmm_4.view(256, 512, 1, 1, 512) ;  bmm_4 = None
        permute_44 = view_82.permute(3, 1, 4, 0, 2) ;  view_82 = None
        view_83 = permute_44.view(1, 512, 512, 256) ;  permute_44 = None
        _to_copy_45 = getitem_87.to(dtype = torch.bfloat16) ;  getitem_87 = None
        _to_copy_46 = getitem_83.to(dtype = torch.bfloat16) 
        t_17 = _to_copy_45.t() ;  _to_copy_45 = None
        view_84 = _to_copy_46.view(262144, 256) ;  _to_copy_46 = None
        mm_17 = torch.mm(view_84,t_17) ;  view_84 = t_17 = None
        view_85 = mm_17.view(1, 512, 512, 512) ;  mm_17 = None
        _to_copy_47 = getitem_89.to(dtype = torch.bfloat16) ;  getitem_89 = None
        _to_copy_48 = getitem_83.to(dtype = torch.bfloat16) 
        t_18 = _to_copy_47.t() ;  _to_copy_47 = None
        view_86 = _to_copy_48.view(262144, 256) ;  _to_copy_48 = None
        mm_18 = torch.mm(view_86,t_18) ;  view_86 = t_18 = None
        view_87 = mm_18.view(1, 512, 512, 512) ;  mm_18 = None
        sigmoid_6 = torch.sigmoid(view_87) ;  view_87 = None
        mul_11 = torch.mul(view_85,sigmoid_6) ;  view_85 = sigmoid_6 = None
        view_88 = mul_11.view(262144, 512) ;  mul_11 = None
        view_89 = view_88.view(1, 512, 512, 512) ;  view_88 = None
        transpose_1 = torch.transpose(bitwise_and_2,1,2) 
        unsqueeze_25 = torch.unsqueeze(transpose_1,3) ;  transpose_1 = None
        clone_5 = torch.clone(unsqueeze_25,memory_format = torch.contiguous_format) ;  unsqueeze_25 = None
        bitwise_not_5 = torch.bitwise_not(clone_5) ;  clone_5 = None
        masked_fill_5 = view_89.masked_fill(bitwise_not_5,0) ;  view_89 = bitwise_not_5 = None
        view_90 = masked_fill_5.view(262144, 512) ;  masked_fill_5 = None
        view_94 = view_90.view(1, 512, 512, 512) 
        split_tensor_16 = torch.split(view_94,256,dim = -1) ;  view_94 = None
        getitem_99 = split_tensor_16[0]
        unsqueeze_28 = torch.unsqueeze(getitem_99,4) ;  getitem_99 = None
        permute_49 = unsqueeze_28.permute(0, 2, 4, 3, 1) ;  unsqueeze_28 = None
        permute_50 = permute_49.permute(3, 1, 4, 0, 2) ;  permute_49 = None
        view_95 = permute_50.view(256, 512, 512) ;  permute_50 = None
        view_96 = view_90.view(1, 512, 512, 512) ;  view_90 = None
        split_tensor_17 = torch.split(view_96,256,dim = -1) ;  view_96 = None
        getitem_102 = split_tensor_17[1];  split_tensor_17 = None
        unsqueeze_29 = torch.unsqueeze(getitem_102,4) ;  getitem_102 = None
        permute_51 = unsqueeze_29.permute(0, 4, 2, 3, 1) ;  unsqueeze_29 = None
        permute_52 = permute_51.permute(3, 4, 0, 2, 1) ;  permute_51 = None
        view_97 = permute_52.view(256, 512, 512) ;  permute_52 = None
        bmm_5 = torch.bmm(view_95,view_97) ;  view_95 = view_97 = None
        view_98 = bmm_5.view(256, 512, 1, 1, 512) ;  bmm_5 = None
        permute_53 = view_98.permute(3, 1, 4, 0, 2) ;  view_98 = None
        view_99 = permute_53.view(1, 512, 512, 256) ;  permute_53 = None
        _to_copy_49 = view_83.to(dtype = torch.float32) ;  view_83 = None
        native_layer_norm_default_9 = (torch.nn.functional.layer_norm(_to_copy_49,[256],None,None,1e-05),) ;  _to_copy_49 = None
        getitem_103 = native_layer_norm_default_9[0]
        _to_copy_50 = view_99.to(dtype = torch.float32) ;  view_99 = None
        native_layer_norm_default_10 = (torch.nn.functional.layer_norm(_to_copy_50,[256],None,None,1e-05),) ;  _to_copy_50 = None
        getitem_106 = native_layer_norm_default_10[0]
        add_12 = torch.add(getitem_103,getitem_106) ;  getitem_103 = getitem_106 = None
        _to_copy_51 = arg33_1.to(dtype = torch.bfloat16) ;  arg33_1 = None
        _to_copy_52 = add_12.to(dtype = torch.bfloat16) ;  add_12 = None
        t_19 = _to_copy_51.t() ;  _to_copy_51 = None
        view_100 = _to_copy_52.view(262144, 256) ;  _to_copy_52 = None
        mm_19 = torch.mm(view_100,t_19) ;  view_100 = t_19 = None
        view_101 = mm_19.view(1, 512, 512, 256) ;  mm_19 = None
        _to_copy_53 = getitem_90.to(dtype = torch.bfloat16) ;  getitem_90 = None
        _to_copy_54 = getitem_83.to(dtype = torch.bfloat16) ;  getitem_83 = None
        t_20 = _to_copy_53.t() ;  _to_copy_53 = None
        view_102 = _to_copy_54.view(262144, 256) ;  _to_copy_54 = None
        mm_20 = torch.mm(view_102,t_20) ;  view_102 = t_20 = None
        view_103 = mm_20.view(1, 512, 512, 256) ;  mm_20 = None
        sigmoid_7 = torch.sigmoid(view_103) ;  view_103 = None
        mul_12 = torch.mul(view_101,sigmoid_7) ;  view_101 = sigmoid_7 = None
        add_13 = torch.add(add_7,mul_12) ;  mul_12 = None
        _to_copy_55 = add_7.to(dtype = torch.float32) 
        native_layer_norm_default_11 = (torch.nn.functional.layer_norm(_to_copy_55,[256],arg36_1,arg37_1,1e-05),) ;  _to_copy_55 = arg36_1 = arg37_1 = None
        getitem_109 = native_layer_norm_default_11[0]
        _to_copy_56 = arg38_1.to(dtype = torch.bfloat16) ;  arg38_1 = None
        _to_copy_57 = getitem_109.to(dtype = torch.bfloat16) ;  getitem_109 = None
        t_21 = _to_copy_56.t() ;  _to_copy_56 = None
        view_104 = _to_copy_57.view(262144, 256) ;  _to_copy_57 = None
        mm_21 = torch.mm(view_104,t_21) ;  view_104 = t_21 = None
        view_105 = mm_21.view(1, 512, 512, 2056) ;  mm_21 = None
        split_with_sizes_default_5 = torch.split(view_105,[2048,8],dim = -1) ;  view_105 = None
        getitem_112 = split_with_sizes_default_5[0]
        getitem_113 = split_with_sizes_default_5[1];  split_with_sizes_default_5 = None
        view_106 = getitem_113.view(1, 512, 512, 2, 4) ;  getitem_113 = None
        permute_54 = view_106.permute(0, 3, 4, 1, 2) ;  view_106 = None
        view_107 = permute_54.view(1, 2, 4, 1, 512, 512) ;  permute_54 = None
        view_108 = bitwise_and_2.view(1, 1, 1, 1, 512, 512) 
        bitwise_not_6 = torch.bitwise_not(view_108) ;  view_108 = None
        masked_fill_6 = view_107.masked_fill(bitwise_not_6,-10000) ;  view_107 = bitwise_not_6 = None
        view_109 = masked_fill_6.view(1, 2, 4, 512, 512) ;  masked_fill_6 = None
        view_110 = view_109.view(8, 1, 512, 512) ;  view_109 = None
        split_tensor_18 = torch.split(getitem_112,1024,dim = -1) ;  getitem_112 = None
        getitem_114 = split_tensor_18[0]
        getitem_115 = split_tensor_18[1];  split_tensor_18 = None
        permute_55 = getitem_115.permute(0, 2, 1, 3) ;  getitem_115 = None
        stack_1 = torch.stack([getitem_114,permute_55]) ;  getitem_114 = permute_55 = None
        view_111 = stack_1.view(2, 1, 512, 512, 4, 4, 64) ;  stack_1 = None
        permute_56 = view_111.permute(4, 1, 0, 5, 2, 3, 6) ;  view_111 = None
        clone_6 = torch.clone(permute_56,memory_format = torch.contiguous_format) ;  permute_56 = None
        _unsafe_view_4 = clone_6.view(4, 8, 512, 512, 64) ;  clone_6 = None
        unbind_int_3 = torch.unbind(_unsafe_view_4) ;  _unsafe_view_4 = None
        getitem_116 = unbind_int_3[0]
        getitem_117 = unbind_int_3[1]
        getitem_118 = unbind_int_3[2]
        getitem_119 = unbind_int_3[3];  unbind_int_3 = None
        split_tensor_19 = torch.split(getitem_116,4) ;  getitem_116 = None
        getitem_120 = split_tensor_19[0]
        getitem_121 = split_tensor_19[1];  split_tensor_19 = None
        split_tensor_20 = torch.split(getitem_117,4) ;  getitem_117 = None
        getitem_122 = split_tensor_20[0]
        getitem_123 = split_tensor_20[1];  split_tensor_20 = None
        split_tensor_21 = torch.split(getitem_118,4) ;  getitem_118 = None
        getitem_124 = split_tensor_21[0]
        getitem_125 = split_tensor_21[1];  split_tensor_21 = None
        split_tensor_22 = torch.split(view_110,4) ;  view_110 = None
        getitem_126 = split_tensor_22[0]
        getitem_127 = split_tensor_22[1];  split_tensor_22 = None
        expand_3 = getitem_126.expand(4, 512, 512, 512) ;  getitem_126 = None
        _scaled_dot_product_efficient_attention_default_3 = (torch.nn.functional.scaled_dot_product_attention(getitem_120,getitem_122,getitem_124,expand_3,False),) ;  getitem_120 = getitem_122 = getitem_124 = expand_3 = None
        getitem_128 = _scaled_dot_product_efficient_attention_default_3[0]
        expand_4 = getitem_127.expand(4, 512, 512, 512) ;  getitem_127 = None
        _scaled_dot_product_efficient_attention_default_4 = (torch.nn.functional.scaled_dot_product_attention(getitem_121,getitem_123,getitem_125,expand_4,False),) ;  getitem_121 = getitem_123 = getitem_125 = expand_4 = None
        getitem_132 = _scaled_dot_product_efficient_attention_default_4[0]
        cat_1 = torch.cat([getitem_128,getitem_132]) ;  getitem_128 = getitem_132 = None
        sigmoid_8 = torch.sigmoid(getitem_119) ;  getitem_119 = None
        mul_13 = torch.mul(cat_1,sigmoid_8) ;  cat_1 = sigmoid_8 = None
        view_112 = mul_13.view(1, 2, 4, 512, 512, 64) ;  mul_13 = None
        permute_57 = view_112.permute(0, 3, 4, 1, 2, 5) ;  view_112 = None
        clone_7 = torch.clone(permute_57,memory_format = torch.contiguous_format) ;  permute_57 = None
        _unsafe_view_5 = clone_7.view(1, 512, 512, 512) ;  clone_7 = None
        _to_copy_58 = arg39_1.to(dtype = torch.bfloat16) ;  arg39_1 = None
        t_22 = _to_copy_58.t() ;  _to_copy_58 = None
        view_113 = _unsafe_view_5.view(262144, 512) ;  _unsafe_view_5 = None
        mm_22 = torch.mm(view_113,t_22) ;  view_113 = t_22 = None
        view_114 = mm_22.view(1, 512, 512, 512) ;  mm_22 = None
        view_115 = view_114.view(1, 512, 512, 2, 4, 64) ;  view_114 = None
        permute_58 = view_115.permute(3, 0, 1, 2, 4, 5) ;  view_115 = None
        view_116 = permute_58.view(2, 1, 512, 512, 256) ;  permute_58 = None
        unbind_int_4 = torch.unbind(view_116) ;  view_116 = None
        getitem_136 = unbind_int_4[0]
        getitem_137 = unbind_int_4[1];  unbind_int_4 = None
        permute_59 = getitem_137.permute(0, 2, 1, 3) ;  getitem_137 = None
        permute_60 = permute_59.permute(0, 2, 1, 3) ;  permute_59 = None
        add_14 = torch.add(getitem_136,permute_60) ;  getitem_136 = permute_60 = None
        add_15 = torch.add(add_13,add_14) ;  add_13 = add_14 = None
        split_tensor_23 = torch.split(add_7,512,dim = -2) 
        getitem_138 = split_tensor_23[0];  split_tensor_23 = None
        _to_copy_59 = getitem_138.to(dtype = torch.float32) ;  getitem_138 = None
        native_layer_norm_default_12 = (torch.nn.functional.layer_norm(_to_copy_59,[256],arg27_1,arg28_1,1e-05),) ;  _to_copy_59 = arg27_1 = arg28_1 = None
        getitem_139 = native_layer_norm_default_12[0]
        _to_copy_60 = arg29_1.to(dtype = torch.bfloat16) ;  arg29_1 = None
        _to_copy_61 = getitem_139.to(dtype = torch.bfloat16) ;  getitem_139 = None
        t_23 = _to_copy_60.t() ;  _to_copy_60 = None
        view_117 = _to_copy_61.view(262144, 256) ;  _to_copy_61 = None
        mm_23 = torch.mm(view_117,t_23) ;  view_117 = t_23 = None
        view_118 = mm_23.view(1, 512, 512, 1024) ;  mm_23 = None
        split_tensor_24 = torch.split(view_118,512,dim = -1) ;  view_118 = None
        getitem_142 = split_tensor_24[0]
        getitem_143 = split_tensor_24[1];  split_tensor_24 = None
        silu_2 = torch.nn.functional.silu(getitem_142) ;  getitem_142 = None
        mul_14 = torch.mul(silu_2,getitem_143) ;  silu_2 = getitem_143 = None
        _to_copy_62 = arg30_1.to(dtype = torch.bfloat16) ;  arg30_1 = None
        t_24 = _to_copy_62.t() ;  _to_copy_62 = None
        view_120 = mul_14.view(262144, 512) ;  mul_14 = None
        mm_24 = torch.mm(view_120,t_24) ;  view_120 = t_24 = None
        view_121 = mm_24.view(1, 512, 512, 256) ;  mm_24 = None
        add_16 = torch.add(add_15,view_121) ;  add_15 = view_121 = None
        _to_copy_63 = add_11.to(dtype = torch.float32) 
        native_layer_norm_default_13 = (torch.nn.functional.layer_norm(_to_copy_63,[384],arg44_1,arg45_1,1e-05),) ;  _to_copy_63 = arg44_1 = arg45_1 = None
        getitem_144 = native_layer_norm_default_13[0]
        _to_copy_64 = add_7.to(dtype = torch.float32) ;  add_7 = None
        native_layer_norm_default_14 = (torch.nn.functional.layer_norm(_to_copy_64,[256],arg46_1,arg47_1,1e-05),) ;  _to_copy_64 = arg46_1 = arg47_1 = None
        getitem_147 = native_layer_norm_default_14[0]
        _to_copy_65 = arg48_1.to(dtype = torch.bfloat16) ;  arg48_1 = None
        _to_copy_66 = getitem_147.to(dtype = torch.bfloat16) ;  getitem_147 = None
        t_25 = _to_copy_65.t() ;  _to_copy_65 = None
        view_122 = _to_copy_66.view(262144, 256) ;  _to_copy_66 = None
        mm_25 = torch.mm(view_122,t_25) ;  view_122 = t_25 = None
        view_123 = mm_25.view(1, 512, 512, 16) ;  mm_25 = None
        permute_61 = view_123.permute(0, 3, 1, 2) ;  view_123 = None
        view_124 = bitwise_and_2.view(1, 1, 512, 512) ;  bitwise_and_2 = None
        bitwise_not_7 = torch.bitwise_not(view_124) ;  view_124 = None
        masked_fill_7 = permute_61.masked_fill(bitwise_not_7,-10000) ;  permute_61 = bitwise_not_7 = None
        _to_copy_67 = getitem_144.to(dtype = torch.bfloat16) ;  getitem_144 = None
        _to_copy_68 = arg50_1.to(dtype = torch.bfloat16) ;  arg50_1 = None
        unsqueeze_30 = torch.unsqueeze(_to_copy_67,3) ;  _to_copy_67 = None
        unsqueeze_31 = torch.unsqueeze(unsqueeze_30,4) ;  unsqueeze_30 = None
        unsqueeze_32 = torch.unsqueeze(unsqueeze_31,5) ;  unsqueeze_31 = None
        permute_62 = unsqueeze_32.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_32 = None
        unsqueeze_33 = torch.unsqueeze(_to_copy_68,4) ;  _to_copy_68 = None
        unsqueeze_34 = torch.unsqueeze(unsqueeze_33,5) ;  unsqueeze_33 = None
        permute_63 = unsqueeze_34.permute(1, 4, 2, 5, 3, 0) ;  unsqueeze_34 = None
        permute_64 = permute_62.permute(3, 5, 0, 1, 2, 4) ;  permute_62 = None
        view_125 = permute_64.view(1, 512, 384) ;  permute_64 = None
        permute_65 = permute_63.permute(5, 0, 1, 2, 4, 3) ;  permute_63 = None
        view_126 = permute_65.view(1, 384, 1536) ;  permute_65 = None
        bmm_6 = torch.bmm(view_125,view_126) ;  view_125 = view_126 = None
        view_127 = bmm_6.view(512, 1, 4, 1, 16, 24) ;  bmm_6 = None
        permute_66 = view_127.permute(2, 3, 4, 0, 5, 1) ;  view_127 = None
        view_128 = permute_66.view(4, 1, 16, 512, 24) ;  permute_66 = None
        unbind_int_5 = torch.unbind(view_128) ;  view_128 = None
        getitem_150 = unbind_int_5[0]
        getitem_151 = unbind_int_5[1]
        getitem_152 = unbind_int_5[2]
        getitem_153 = unbind_int_5[3];  unbind_int_5 = None
        view_129 = arg49_1.view(1, 16, 1, 24) ;  arg49_1 = None
        add_17 = torch.add(getitem_150,view_129) ;  getitem_150 = view_129 = None
        _to_copy_69 = add_17.to(dtype = torch.bfloat16) ;  add_17 = None
        expand_5 = masked_fill_7.expand(1, 16, 512, 512) ;  masked_fill_7 = None
        _scaled_dot_product_efficient_attention_default_5 = (torch.nn.functional.scaled_dot_product_attention(_to_copy_69,getitem_151,getitem_152,expand_5,False),) ;  _to_copy_69 = getitem_151 = getitem_152 = expand_5 = None
        getitem_154 = _scaled_dot_product_efficient_attention_default_5[0]
        add_18 = torch.add(getitem_153,1) ;  getitem_153 = None
        sigmoid_9 = torch.sigmoid(add_18) ;  add_18 = None
        mul_15 = torch.mul(getitem_154,sigmoid_9) ;  getitem_154 = sigmoid_9 = None
        _to_copy_70 = arg51_1.to(dtype = torch.bfloat16) ;  arg51_1 = None
        unsqueeze_35 = torch.unsqueeze(mul_15,4) ;  mul_15 = None
        permute_67 = unsqueeze_35.permute(0, 2, 4, 3, 1) ;  unsqueeze_35 = None
        unsqueeze_36 = torch.unsqueeze(_to_copy_70,3) ;  _to_copy_70 = None
        unsqueeze_37 = torch.unsqueeze(unsqueeze_36,4) ;  unsqueeze_36 = None
        permute_68 = unsqueeze_37.permute(3, 4, 2, 1, 0) ;  unsqueeze_37 = None
        permute_69 = permute_67.permute(1, 3, 4, 0, 2) ;  permute_67 = None
        clone_8 = torch.clone(permute_69,memory_format = torch.contiguous_format) ;  permute_69 = None
        _unsafe_view_6 = clone_8.view(1, 512, 384) ;  clone_8 = None
        permute_70 = permute_68.permute(3, 4, 0, 2, 1) ;  permute_68 = None
        clone_9 = torch.clone(permute_70,memory_format = torch.contiguous_format) ;  permute_70 = None
        _unsafe_view_7 = clone_9.view(1, 384, 384) ;  clone_9 = None
        bmm_7 = torch.bmm(_unsafe_view_6,_unsafe_view_7) ;  _unsafe_view_6 = _unsafe_view_7 = None
        view_130 = bmm_7.view(512, 1, 1, 1, 384) ;  bmm_7 = None
        permute_71 = view_130.permute(3, 0, 4, 1, 2) ;  view_130 = None
        view_131 = permute_71.view(1, 512, 384) ;  permute_71 = None
        unsqueeze_38 = torch.unsqueeze(arg109_1,-1) 
        mul_16 = torch.mul(view_131,unsqueeze_38) ;  view_131 = unsqueeze_38 = None
        add_19 = torch.add(add_11,mul_16) ;  mul_16 = None
        split_tensor_25 = torch.split(add_11,512,dim = -2) ;  add_11 = None
        getitem_158 = split_tensor_25[0];  split_tensor_25 = None
        _to_copy_71 = getitem_158.to(dtype = torch.float32) ;  getitem_158 = None
        native_layer_norm_default_15 = (torch.nn.functional.layer_norm(_to_copy_71,[384],arg40_1,arg41_1,1e-05),) ;  _to_copy_71 = arg40_1 = arg41_1 = None
        getitem_159 = native_layer_norm_default_15[0]
        _to_copy_72 = arg42_1.to(dtype = torch.bfloat16) ;  arg42_1 = None
        _to_copy_73 = getitem_159.to(dtype = torch.bfloat16) ;  getitem_159 = None
        t_26 = _to_copy_72.t() ;  _to_copy_72 = None
        view_132 = _to_copy_73.view(512, 384) ;  _to_copy_73 = None
        mm_26 = torch.mm(view_132,t_26) ;  view_132 = t_26 = None
        view_133 = mm_26.view(1, 512, 1536) ;  mm_26 = None
        split_tensor_26 = torch.split(view_133,768,dim = -1) ;  view_133 = None
        getitem_162 = split_tensor_26[0]
        getitem_163 = split_tensor_26[1];  split_tensor_26 = None
        silu_3 = torch.nn.functional.silu(getitem_162) ;  getitem_162 = None
        mul_17 = torch.mul(silu_3,getitem_163) ;  silu_3 = getitem_163 = None
        _to_copy_74 = arg43_1.to(dtype = torch.bfloat16) ;  arg43_1 = None
        t_27 = _to_copy_74.t() ;  _to_copy_74 = None
        view_135 = mul_17.view(512, 768) ;  mul_17 = None
        mm_27 = torch.mm(view_135,t_27) ;  view_135 = t_27 = None
        view_136 = mm_27.view(1, 512, 384) ;  mm_27 = None
        add_20 = torch.add(add_19,view_136) ;  add_19 = view_136 = None
        view_137 = arg109_1.view(1, 512, 1) 
        view_138 = arg109_1.view(1, 1, 512) 
        bitwise_and_3 = torch.bitwise_and(view_137,view_138) ;  view_137 = view_138 = None
        _to_copy_75 = add_16.to(dtype = torch.float32) 
        native_layer_norm_default_16 = (torch.nn.functional.layer_norm(_to_copy_75,[256],arg56_1,arg57_1,1e-05),) ;  _to_copy_75 = arg56_1 = arg57_1 = None
        getitem_164 = native_layer_norm_default_16[0]
        split_with_sizes_default_6 = torch.split(arg59_1,[512,512]) ;  arg59_1 = None
        getitem_167 = split_with_sizes_default_6[0]
        getitem_168 = split_with_sizes_default_6[1];  split_with_sizes_default_6 = None
        split_with_sizes_default_7 = torch.split(arg60_1,[512,512,256]) ;  arg60_1 = None
        getitem_169 = split_with_sizes_default_7[0]
        getitem_170 = split_with_sizes_default_7[1]
        getitem_171 = split_with_sizes_default_7[2];  split_with_sizes_default_7 = None
        _to_copy_76 = getitem_167.to(dtype = torch.bfloat16) ;  getitem_167 = None
        _to_copy_77 = getitem_164.to(dtype = torch.bfloat16) 
        t_28 = _to_copy_76.t() ;  _to_copy_76 = None
        view_139 = _to_copy_77.view(262144, 256) ;  _to_copy_77 = None
        mm_28 = torch.mm(view_139,t_28) ;  view_139 = t_28 = None
        view_140 = mm_28.view(1, 512, 512, 512) ;  mm_28 = None
        _to_copy_78 = getitem_169.to(dtype = torch.bfloat16) ;  getitem_169 = None
        _to_copy_79 = getitem_164.to(dtype = torch.bfloat16) 
        t_29 = _to_copy_78.t() ;  _to_copy_78 = None
        view_141 = _to_copy_79.view(262144, 256) ;  _to_copy_79 = None
        mm_29 = torch.mm(view_141,t_29) ;  view_141 = t_29 = None
        view_142 = mm_29.view(1, 512, 512, 512) ;  mm_29 = None
        sigmoid_10 = torch.sigmoid(view_142) ;  view_142 = None
        mul_18 = torch.mul(view_140,sigmoid_10) ;  view_140 = sigmoid_10 = None
        unsqueeze_39 = torch.unsqueeze(bitwise_and_3,3) 
        bitwise_not_8 = torch.bitwise_not(unsqueeze_39) ;  unsqueeze_39 = None
        masked_fill_8 = mul_18.masked_fill(bitwise_not_8,0) ;  mul_18 = bitwise_not_8 = None
        split_tensor_27 = torch.split(masked_fill_8,256,dim = -1) 
        getitem_174 = split_tensor_27[0]
        unsqueeze_42 = torch.unsqueeze(getitem_174,4) ;  getitem_174 = None
        permute_76 = unsqueeze_42.permute(0, 1, 4, 3, 2) ;  unsqueeze_42 = None
        permute_77 = permute_76.permute(3, 1, 4, 0, 2) ;  permute_76 = None
        view_145 = permute_77.view(256, 512, 512) ;  permute_77 = None
        split_tensor_28 = torch.split(masked_fill_8,256,dim = -1) ;  masked_fill_8 = None
        getitem_177 = split_tensor_28[1];  split_tensor_28 = None
        unsqueeze_43 = torch.unsqueeze(getitem_177,4) ;  getitem_177 = None
        permute_78 = unsqueeze_43.permute(0, 4, 1, 3, 2) ;  unsqueeze_43 = None
        permute_79 = permute_78.permute(3, 4, 0, 2, 1) ;  permute_78 = None
        view_146 = permute_79.view(256, 512, 512) ;  permute_79 = None
        bmm_8 = torch.bmm(view_145,view_146) ;  view_145 = view_146 = None
        view_147 = bmm_8.view(256, 512, 1, 1, 512) ;  bmm_8 = None
        permute_80 = view_147.permute(3, 1, 4, 0, 2) ;  view_147 = None
        view_148 = permute_80.view(1, 512, 512, 256) ;  permute_80 = None
        _to_copy_80 = getitem_168.to(dtype = torch.bfloat16) ;  getitem_168 = None
        _to_copy_81 = getitem_164.to(dtype = torch.bfloat16) 
        t_30 = _to_copy_80.t() ;  _to_copy_80 = None
        view_149 = _to_copy_81.view(262144, 256) ;  _to_copy_81 = None
        mm_30 = torch.mm(view_149,t_30) ;  view_149 = t_30 = None
        view_150 = mm_30.view(1, 512, 512, 512) ;  mm_30 = None
        _to_copy_82 = getitem_170.to(dtype = torch.bfloat16) ;  getitem_170 = None
        _to_copy_83 = getitem_164.to(dtype = torch.bfloat16) 
        t_31 = _to_copy_82.t() ;  _to_copy_82 = None
        view_151 = _to_copy_83.view(262144, 256) ;  _to_copy_83 = None
        mm_31 = torch.mm(view_151,t_31) ;  view_151 = t_31 = None
        view_152 = mm_31.view(1, 512, 512, 512) ;  mm_31 = None
        sigmoid_11 = torch.sigmoid(view_152) ;  view_152 = None
        mul_19 = torch.mul(view_150,sigmoid_11) ;  view_150 = sigmoid_11 = None
        view_153 = mul_19.view(262144, 512) ;  mul_19 = None
        view_154 = view_153.view(1, 512, 512, 512) ;  view_153 = None
        transpose_2 = torch.transpose(bitwise_and_3,1,2) 
        unsqueeze_44 = torch.unsqueeze(transpose_2,3) ;  transpose_2 = None
        clone_10 = torch.clone(unsqueeze_44,memory_format = torch.contiguous_format) ;  unsqueeze_44 = None
        bitwise_not_9 = torch.bitwise_not(clone_10) ;  clone_10 = None
        masked_fill_9 = view_154.masked_fill(bitwise_not_9,0) ;  view_154 = bitwise_not_9 = None
        view_155 = masked_fill_9.view(262144, 512) ;  masked_fill_9 = None
        view_159 = view_155.view(1, 512, 512, 512) 
        split_tensor_29 = torch.split(view_159,256,dim = -1) ;  view_159 = None
        getitem_180 = split_tensor_29[0]
        unsqueeze_47 = torch.unsqueeze(getitem_180,4) ;  getitem_180 = None
        permute_85 = unsqueeze_47.permute(0, 2, 4, 3, 1) ;  unsqueeze_47 = None
        permute_86 = permute_85.permute(3, 1, 4, 0, 2) ;  permute_85 = None
        view_160 = permute_86.view(256, 512, 512) ;  permute_86 = None
        view_161 = view_155.view(1, 512, 512, 512) ;  view_155 = None
        split_tensor_30 = torch.split(view_161,256,dim = -1) ;  view_161 = None
        getitem_183 = split_tensor_30[1];  split_tensor_30 = None
        unsqueeze_48 = torch.unsqueeze(getitem_183,4) ;  getitem_183 = None
        permute_87 = unsqueeze_48.permute(0, 4, 2, 3, 1) ;  unsqueeze_48 = None
        permute_88 = permute_87.permute(3, 4, 0, 2, 1) ;  permute_87 = None
        view_162 = permute_88.view(256, 512, 512) ;  permute_88 = None
        bmm_9 = torch.bmm(view_160,view_162) ;  view_160 = view_162 = None
        view_163 = bmm_9.view(256, 512, 1, 1, 512) ;  bmm_9 = None
        permute_89 = view_163.permute(3, 1, 4, 0, 2) ;  view_163 = None
        view_164 = permute_89.view(1, 512, 512, 256) ;  permute_89 = None
        _to_copy_84 = view_148.to(dtype = torch.float32) ;  view_148 = None
        native_layer_norm_default_17 = (torch.nn.functional.layer_norm(_to_copy_84,[256],None,None,1e-05),) ;  _to_copy_84 = None
        getitem_184 = native_layer_norm_default_17[0]
        _to_copy_85 = view_164.to(dtype = torch.float32) ;  view_164 = None
        native_layer_norm_default_18 = (torch.nn.functional.layer_norm(_to_copy_85,[256],None,None,1e-05),) ;  _to_copy_85 = None
        getitem_187 = native_layer_norm_default_18[0]
        add_21 = torch.add(getitem_184,getitem_187) ;  getitem_184 = getitem_187 = None
        _to_copy_86 = arg58_1.to(dtype = torch.bfloat16) ;  arg58_1 = None
        _to_copy_87 = add_21.to(dtype = torch.bfloat16) ;  add_21 = None
        t_32 = _to_copy_86.t() ;  _to_copy_86 = None
        view_165 = _to_copy_87.view(262144, 256) ;  _to_copy_87 = None
        mm_32 = torch.mm(view_165,t_32) ;  view_165 = t_32 = None
        view_166 = mm_32.view(1, 512, 512, 256) ;  mm_32 = None
        _to_copy_88 = getitem_171.to(dtype = torch.bfloat16) ;  getitem_171 = None
        _to_copy_89 = getitem_164.to(dtype = torch.bfloat16) ;  getitem_164 = None
        t_33 = _to_copy_88.t() ;  _to_copy_88 = None
        view_167 = _to_copy_89.view(262144, 256) ;  _to_copy_89 = None
        mm_33 = torch.mm(view_167,t_33) ;  view_167 = t_33 = None
        view_168 = mm_33.view(1, 512, 512, 256) ;  mm_33 = None
        sigmoid_12 = torch.sigmoid(view_168) ;  view_168 = None
        mul_20 = torch.mul(view_166,sigmoid_12) ;  view_166 = sigmoid_12 = None
        add_22 = torch.add(add_16,mul_20) ;  mul_20 = None
        _to_copy_90 = add_16.to(dtype = torch.float32) 
        native_layer_norm_default_19 = (torch.nn.functional.layer_norm(_to_copy_90,[256],arg61_1,arg62_1,1e-05),) ;  _to_copy_90 = arg61_1 = arg62_1 = None
        getitem_190 = native_layer_norm_default_19[0]
        _to_copy_91 = arg63_1.to(dtype = torch.bfloat16) ;  arg63_1 = None
        _to_copy_92 = getitem_190.to(dtype = torch.bfloat16) ;  getitem_190 = None
        t_34 = _to_copy_91.t() ;  _to_copy_91 = None
        view_169 = _to_copy_92.view(262144, 256) ;  _to_copy_92 = None
        mm_34 = torch.mm(view_169,t_34) ;  view_169 = t_34 = None
        view_170 = mm_34.view(1, 512, 512, 2056) ;  mm_34 = None
        split_with_sizes_default_8 = torch.split(view_170,[2048,8],dim = -1) ;  view_170 = None
        getitem_193 = split_with_sizes_default_8[0]
        getitem_194 = split_with_sizes_default_8[1];  split_with_sizes_default_8 = None
        view_171 = getitem_194.view(1, 512, 512, 2, 4) ;  getitem_194 = None
        permute_90 = view_171.permute(0, 3, 4, 1, 2) ;  view_171 = None
        view_172 = permute_90.view(1, 2, 4, 1, 512, 512) ;  permute_90 = None
        view_173 = bitwise_and_3.view(1, 1, 1, 1, 512, 512) 
        bitwise_not_10 = torch.bitwise_not(view_173) ;  view_173 = None
        masked_fill_10 = view_172.masked_fill(bitwise_not_10,-10000) ;  view_172 = bitwise_not_10 = None
        view_174 = masked_fill_10.view(1, 2, 4, 512, 512) ;  masked_fill_10 = None
        view_175 = view_174.view(8, 1, 512, 512) ;  view_174 = None
        split_tensor_31 = torch.split(getitem_193,1024,dim = -1) ;  getitem_193 = None
        getitem_195 = split_tensor_31[0]
        getitem_196 = split_tensor_31[1];  split_tensor_31 = None
        permute_91 = getitem_196.permute(0, 2, 1, 3) ;  getitem_196 = None
        stack_2 = torch.stack([getitem_195,permute_91]) ;  getitem_195 = permute_91 = None
        view_176 = stack_2.view(2, 1, 512, 512, 4, 4, 64) ;  stack_2 = None
        permute_92 = view_176.permute(4, 1, 0, 5, 2, 3, 6) ;  view_176 = None
        clone_11 = torch.clone(permute_92,memory_format = torch.contiguous_format) ;  permute_92 = None
        _unsafe_view_8 = clone_11.view(4, 8, 512, 512, 64) ;  clone_11 = None
        unbind_int_6 = torch.unbind(_unsafe_view_8) ;  _unsafe_view_8 = None
        getitem_197 = unbind_int_6[0]
        getitem_198 = unbind_int_6[1]
        getitem_199 = unbind_int_6[2]
        getitem_200 = unbind_int_6[3];  unbind_int_6 = None
        split_tensor_32 = torch.split(getitem_197,4) ;  getitem_197 = None
        getitem_201 = split_tensor_32[0]
        getitem_202 = split_tensor_32[1];  split_tensor_32 = None
        split_tensor_33 = torch.split(getitem_198,4) ;  getitem_198 = None
        getitem_203 = split_tensor_33[0]
        getitem_204 = split_tensor_33[1];  split_tensor_33 = None
        split_tensor_34 = torch.split(getitem_199,4) ;  getitem_199 = None
        getitem_205 = split_tensor_34[0]
        getitem_206 = split_tensor_34[1];  split_tensor_34 = None
        split_tensor_35 = torch.split(view_175,4) ;  view_175 = None
        getitem_207 = split_tensor_35[0]
        getitem_208 = split_tensor_35[1];  split_tensor_35 = None
        expand_6 = getitem_207.expand(4, 512, 512, 512) ;  getitem_207 = None
        _scaled_dot_product_efficient_attention_default_6 = (torch.nn.functional.scaled_dot_product_attention(getitem_201,getitem_203,getitem_205,expand_6,False),) ;  getitem_201 = getitem_203 = getitem_205 = expand_6 = None
        getitem_209 = _scaled_dot_product_efficient_attention_default_6[0]
        expand_7 = getitem_208.expand(4, 512, 512, 512) ;  getitem_208 = None
        _scaled_dot_product_efficient_attention_default_7 = (torch.nn.functional.scaled_dot_product_attention(getitem_202,getitem_204,getitem_206,expand_7,False),) ;  getitem_202 = getitem_204 = getitem_206 = expand_7 = None
        getitem_213 = _scaled_dot_product_efficient_attention_default_7[0]
        cat_2 = torch.cat([getitem_209,getitem_213]) ;  getitem_209 = getitem_213 = None
        sigmoid_13 = torch.sigmoid(getitem_200) ;  getitem_200 = None
        mul_21 = torch.mul(cat_2,sigmoid_13) ;  cat_2 = sigmoid_13 = None
        view_177 = mul_21.view(1, 2, 4, 512, 512, 64) ;  mul_21 = None
        permute_93 = view_177.permute(0, 3, 4, 1, 2, 5) ;  view_177 = None
        clone_12 = torch.clone(permute_93,memory_format = torch.contiguous_format) ;  permute_93 = None
        _unsafe_view_9 = clone_12.view(1, 512, 512, 512) ;  clone_12 = None
        _to_copy_93 = arg64_1.to(dtype = torch.bfloat16) ;  arg64_1 = None
        t_35 = _to_copy_93.t() ;  _to_copy_93 = None
        view_178 = _unsafe_view_9.view(262144, 512) ;  _unsafe_view_9 = None
        mm_35 = torch.mm(view_178,t_35) ;  view_178 = t_35 = None
        view_179 = mm_35.view(1, 512, 512, 512) ;  mm_35 = None
        view_180 = view_179.view(1, 512, 512, 2, 4, 64) ;  view_179 = None
        permute_94 = view_180.permute(3, 0, 1, 2, 4, 5) ;  view_180 = None
        view_181 = permute_94.view(2, 1, 512, 512, 256) ;  permute_94 = None
        unbind_int_7 = torch.unbind(view_181) ;  view_181 = None
        getitem_217 = unbind_int_7[0]
        getitem_218 = unbind_int_7[1];  unbind_int_7 = None
        permute_95 = getitem_218.permute(0, 2, 1, 3) ;  getitem_218 = None
        permute_96 = permute_95.permute(0, 2, 1, 3) ;  permute_95 = None
        add_23 = torch.add(getitem_217,permute_96) ;  getitem_217 = permute_96 = None
        add_24 = torch.add(add_22,add_23) ;  add_22 = add_23 = None
        split_tensor_36 = torch.split(add_16,512,dim = -2) 
        getitem_219 = split_tensor_36[0];  split_tensor_36 = None
        _to_copy_94 = getitem_219.to(dtype = torch.float32) ;  getitem_219 = None
        native_layer_norm_default_20 = (torch.nn.functional.layer_norm(_to_copy_94,[256],arg52_1,arg53_1,1e-05),) ;  _to_copy_94 = arg52_1 = arg53_1 = None
        getitem_220 = native_layer_norm_default_20[0]
        _to_copy_95 = arg54_1.to(dtype = torch.bfloat16) ;  arg54_1 = None
        _to_copy_96 = getitem_220.to(dtype = torch.bfloat16) ;  getitem_220 = None
        t_36 = _to_copy_95.t() ;  _to_copy_95 = None
        view_182 = _to_copy_96.view(262144, 256) ;  _to_copy_96 = None
        mm_36 = torch.mm(view_182,t_36) ;  view_182 = t_36 = None
        view_183 = mm_36.view(1, 512, 512, 1024) ;  mm_36 = None
        split_tensor_37 = torch.split(view_183,512,dim = -1) ;  view_183 = None
        getitem_223 = split_tensor_37[0]
        getitem_224 = split_tensor_37[1];  split_tensor_37 = None
        silu_4 = torch.nn.functional.silu(getitem_223) ;  getitem_223 = None
        mul_22 = torch.mul(silu_4,getitem_224) ;  silu_4 = getitem_224 = None
        _to_copy_97 = arg55_1.to(dtype = torch.bfloat16) ;  arg55_1 = None
        t_37 = _to_copy_97.t() ;  _to_copy_97 = None
        view_185 = mul_22.view(262144, 512) ;  mul_22 = None
        mm_37 = torch.mm(view_185,t_37) ;  view_185 = t_37 = None
        view_186 = mm_37.view(1, 512, 512, 256) ;  mm_37 = None
        add_25 = torch.add(add_24,view_186) ;  add_24 = view_186 = None
        _to_copy_98 = add_20.to(dtype = torch.float32) 
        native_layer_norm_default_21 = (torch.nn.functional.layer_norm(_to_copy_98,[384],arg69_1,arg70_1,1e-05),) ;  _to_copy_98 = arg69_1 = arg70_1 = None
        getitem_225 = native_layer_norm_default_21[0]
        _to_copy_99 = add_16.to(dtype = torch.float32) ;  add_16 = None
        native_layer_norm_default_22 = (torch.nn.functional.layer_norm(_to_copy_99,[256],arg71_1,arg72_1,1e-05),) ;  _to_copy_99 = arg71_1 = arg72_1 = None
        getitem_228 = native_layer_norm_default_22[0]
        _to_copy_100 = arg73_1.to(dtype = torch.bfloat16) ;  arg73_1 = None
        _to_copy_101 = getitem_228.to(dtype = torch.bfloat16) ;  getitem_228 = None
        t_38 = _to_copy_100.t() ;  _to_copy_100 = None
        view_187 = _to_copy_101.view(262144, 256) ;  _to_copy_101 = None
        mm_38 = torch.mm(view_187,t_38) ;  view_187 = t_38 = None
        view_188 = mm_38.view(1, 512, 512, 16) ;  mm_38 = None
        permute_97 = view_188.permute(0, 3, 1, 2) ;  view_188 = None
        view_189 = bitwise_and_3.view(1, 1, 512, 512) ;  bitwise_and_3 = None
        bitwise_not_11 = torch.bitwise_not(view_189) ;  view_189 = None
        masked_fill_11 = permute_97.masked_fill(bitwise_not_11,-10000) ;  permute_97 = bitwise_not_11 = None
        _to_copy_102 = getitem_225.to(dtype = torch.bfloat16) ;  getitem_225 = None
        _to_copy_103 = arg75_1.to(dtype = torch.bfloat16) ;  arg75_1 = None
        unsqueeze_49 = torch.unsqueeze(_to_copy_102,3) ;  _to_copy_102 = None
        unsqueeze_50 = torch.unsqueeze(unsqueeze_49,4) ;  unsqueeze_49 = None
        unsqueeze_51 = torch.unsqueeze(unsqueeze_50,5) ;  unsqueeze_50 = None
        permute_98 = unsqueeze_51.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_51 = None
        unsqueeze_52 = torch.unsqueeze(_to_copy_103,4) ;  _to_copy_103 = None
        unsqueeze_53 = torch.unsqueeze(unsqueeze_52,5) ;  unsqueeze_52 = None
        permute_99 = unsqueeze_53.permute(1, 4, 2, 5, 3, 0) ;  unsqueeze_53 = None
        permute_100 = permute_98.permute(3, 5, 0, 1, 2, 4) ;  permute_98 = None
        view_190 = permute_100.view(1, 512, 384) ;  permute_100 = None
        permute_101 = permute_99.permute(5, 0, 1, 2, 4, 3) ;  permute_99 = None
        view_191 = permute_101.view(1, 384, 1536) ;  permute_101 = None
        bmm_10 = torch.bmm(view_190,view_191) ;  view_190 = view_191 = None
        view_192 = bmm_10.view(512, 1, 4, 1, 16, 24) ;  bmm_10 = None
        permute_102 = view_192.permute(2, 3, 4, 0, 5, 1) ;  view_192 = None
        view_193 = permute_102.view(4, 1, 16, 512, 24) ;  permute_102 = None
        unbind_int_8 = torch.unbind(view_193) ;  view_193 = None
        getitem_231 = unbind_int_8[0]
        getitem_232 = unbind_int_8[1]
        getitem_233 = unbind_int_8[2]
        getitem_234 = unbind_int_8[3];  unbind_int_8 = None
        view_194 = arg74_1.view(1, 16, 1, 24) ;  arg74_1 = None
        add_26 = torch.add(getitem_231,view_194) ;  getitem_231 = view_194 = None
        _to_copy_104 = add_26.to(dtype = torch.bfloat16) ;  add_26 = None
        expand_8 = masked_fill_11.expand(1, 16, 512, 512) ;  masked_fill_11 = None
        _scaled_dot_product_efficient_attention_default_8 = (torch.nn.functional.scaled_dot_product_attention(_to_copy_104,getitem_232,getitem_233,expand_8,False),) ;  _to_copy_104 = getitem_232 = getitem_233 = expand_8 = None
        getitem_235 = _scaled_dot_product_efficient_attention_default_8[0]
        add_27 = torch.add(getitem_234,1) ;  getitem_234 = None
        sigmoid_14 = torch.sigmoid(add_27) ;  add_27 = None
        mul_23 = torch.mul(getitem_235,sigmoid_14) ;  getitem_235 = sigmoid_14 = None
        _to_copy_105 = arg76_1.to(dtype = torch.bfloat16) ;  arg76_1 = None
        unsqueeze_54 = torch.unsqueeze(mul_23,4) ;  mul_23 = None
        permute_103 = unsqueeze_54.permute(0, 2, 4, 3, 1) ;  unsqueeze_54 = None
        unsqueeze_55 = torch.unsqueeze(_to_copy_105,3) ;  _to_copy_105 = None
        unsqueeze_56 = torch.unsqueeze(unsqueeze_55,4) ;  unsqueeze_55 = None
        permute_104 = unsqueeze_56.permute(3, 4, 2, 1, 0) ;  unsqueeze_56 = None
        permute_105 = permute_103.permute(1, 3, 4, 0, 2) ;  permute_103 = None
        clone_13 = torch.clone(permute_105,memory_format = torch.contiguous_format) ;  permute_105 = None
        _unsafe_view_10 = clone_13.view(1, 512, 384) ;  clone_13 = None
        permute_106 = permute_104.permute(3, 4, 0, 2, 1) ;  permute_104 = None
        clone_14 = torch.clone(permute_106,memory_format = torch.contiguous_format) ;  permute_106 = None
        _unsafe_view_11 = clone_14.view(1, 384, 384) ;  clone_14 = None
        bmm_11 = torch.bmm(_unsafe_view_10,_unsafe_view_11) ;  _unsafe_view_10 = _unsafe_view_11 = None
        view_195 = bmm_11.view(512, 1, 1, 1, 384) ;  bmm_11 = None
        permute_107 = view_195.permute(3, 0, 4, 1, 2) ;  view_195 = None
        view_196 = permute_107.view(1, 512, 384) ;  permute_107 = None
        unsqueeze_57 = torch.unsqueeze(arg109_1,-1) 
        mul_24 = torch.mul(view_196,unsqueeze_57) ;  view_196 = unsqueeze_57 = None
        add_28 = torch.add(add_20,mul_24) ;  mul_24 = None
        split_tensor_38 = torch.split(add_20,512,dim = -2) ;  add_20 = None
        getitem_239 = split_tensor_38[0];  split_tensor_38 = None
        _to_copy_106 = getitem_239.to(dtype = torch.float32) ;  getitem_239 = None
        native_layer_norm_default_23 = (torch.nn.functional.layer_norm(_to_copy_106,[384],arg65_1,arg66_1,1e-05),) ;  _to_copy_106 = arg65_1 = arg66_1 = None
        getitem_240 = native_layer_norm_default_23[0]
        _to_copy_107 = arg67_1.to(dtype = torch.bfloat16) ;  arg67_1 = None
        _to_copy_108 = getitem_240.to(dtype = torch.bfloat16) ;  getitem_240 = None
        t_39 = _to_copy_107.t() ;  _to_copy_107 = None
        view_197 = _to_copy_108.view(512, 384) ;  _to_copy_108 = None
        mm_39 = torch.mm(view_197,t_39) ;  view_197 = t_39 = None
        view_198 = mm_39.view(1, 512, 1536) ;  mm_39 = None
        split_tensor_39 = torch.split(view_198,768,dim = -1) ;  view_198 = None
        getitem_243 = split_tensor_39[0]
        getitem_244 = split_tensor_39[1];  split_tensor_39 = None
        silu_5 = torch.nn.functional.silu(getitem_243) ;  getitem_243 = None
        mul_25 = torch.mul(silu_5,getitem_244) ;  silu_5 = getitem_244 = None
        _to_copy_109 = arg68_1.to(dtype = torch.bfloat16) ;  arg68_1 = None
        t_40 = _to_copy_109.t() ;  _to_copy_109 = None
        view_200 = mul_25.view(512, 768) ;  mul_25 = None
        mm_40 = torch.mm(view_200,t_40) ;  view_200 = t_40 = None
        view_201 = mm_40.view(1, 512, 384) ;  mm_40 = None
        add_29 = torch.add(add_28,view_201) ;  add_28 = view_201 = None
        view_202 = arg109_1.view(1, 512, 1) 
        view_203 = arg109_1.view(1, 1, 512) 
        bitwise_and_4 = torch.bitwise_and(view_202,view_203) ;  view_202 = view_203 = None
        _to_copy_110 = add_25.to(dtype = torch.float32) 
        native_layer_norm_default_24 = (torch.nn.functional.layer_norm(_to_copy_110,[256],arg81_1,arg82_1,1e-05),) ;  _to_copy_110 = arg81_1 = arg82_1 = None
        getitem_245 = native_layer_norm_default_24[0]
        split_with_sizes_default_9 = torch.split(arg84_1,[512,512]) ;  arg84_1 = None
        getitem_248 = split_with_sizes_default_9[0]
        getitem_249 = split_with_sizes_default_9[1];  split_with_sizes_default_9 = None
        split_with_sizes_default_10 = torch.split(arg85_1,[512,512,256]) ;  arg85_1 = None
        getitem_250 = split_with_sizes_default_10[0]
        getitem_251 = split_with_sizes_default_10[1]
        getitem_252 = split_with_sizes_default_10[2];  split_with_sizes_default_10 = None
        _to_copy_111 = getitem_248.to(dtype = torch.bfloat16) ;  getitem_248 = None
        _to_copy_112 = getitem_245.to(dtype = torch.bfloat16) 
        t_41 = _to_copy_111.t() ;  _to_copy_111 = None
        view_204 = _to_copy_112.view(262144, 256) ;  _to_copy_112 = None
        mm_41 = torch.mm(view_204,t_41) ;  view_204 = t_41 = None
        view_205 = mm_41.view(1, 512, 512, 512) ;  mm_41 = None
        _to_copy_113 = getitem_250.to(dtype = torch.bfloat16) ;  getitem_250 = None
        _to_copy_114 = getitem_245.to(dtype = torch.bfloat16) 
        t_42 = _to_copy_113.t() ;  _to_copy_113 = None
        view_206 = _to_copy_114.view(262144, 256) ;  _to_copy_114 = None
        mm_42 = torch.mm(view_206,t_42) ;  view_206 = t_42 = None
        view_207 = mm_42.view(1, 512, 512, 512) ;  mm_42 = None
        sigmoid_15 = torch.sigmoid(view_207) ;  view_207 = None
        mul_26 = torch.mul(view_205,sigmoid_15) ;  view_205 = sigmoid_15 = None
        unsqueeze_58 = torch.unsqueeze(bitwise_and_4,3) 
        bitwise_not_12 = torch.bitwise_not(unsqueeze_58) ;  unsqueeze_58 = None
        masked_fill_12 = mul_26.masked_fill(bitwise_not_12,0) ;  mul_26 = bitwise_not_12 = None
        split_tensor_40 = torch.split(masked_fill_12,256,dim = -1) 
        getitem_255 = split_tensor_40[0]
        unsqueeze_61 = torch.unsqueeze(getitem_255,4) ;  getitem_255 = None
        permute_112 = unsqueeze_61.permute(0, 1, 4, 3, 2) ;  unsqueeze_61 = None
        permute_113 = permute_112.permute(3, 1, 4, 0, 2) ;  permute_112 = None
        view_210 = permute_113.view(256, 512, 512) ;  permute_113 = None
        split_tensor_41 = torch.split(masked_fill_12,256,dim = -1) ;  masked_fill_12 = None
        getitem_258 = split_tensor_41[1];  split_tensor_41 = None
        unsqueeze_62 = torch.unsqueeze(getitem_258,4) ;  getitem_258 = None
        permute_114 = unsqueeze_62.permute(0, 4, 1, 3, 2) ;  unsqueeze_62 = None
        permute_115 = permute_114.permute(3, 4, 0, 2, 1) ;  permute_114 = None
        view_211 = permute_115.view(256, 512, 512) ;  permute_115 = None
        bmm_12 = torch.bmm(view_210,view_211) ;  view_210 = view_211 = None
        view_212 = bmm_12.view(256, 512, 1, 1, 512) ;  bmm_12 = None
        permute_116 = view_212.permute(3, 1, 4, 0, 2) ;  view_212 = None
        view_213 = permute_116.view(1, 512, 512, 256) ;  permute_116 = None
        _to_copy_115 = getitem_249.to(dtype = torch.bfloat16) ;  getitem_249 = None
        _to_copy_116 = getitem_245.to(dtype = torch.bfloat16) 
        t_43 = _to_copy_115.t() ;  _to_copy_115 = None
        view_214 = _to_copy_116.view(262144, 256) ;  _to_copy_116 = None
        mm_43 = torch.mm(view_214,t_43) ;  view_214 = t_43 = None
        view_215 = mm_43.view(1, 512, 512, 512) ;  mm_43 = None
        _to_copy_117 = getitem_251.to(dtype = torch.bfloat16) ;  getitem_251 = None
        _to_copy_118 = getitem_245.to(dtype = torch.bfloat16) 
        t_44 = _to_copy_117.t() ;  _to_copy_117 = None
        view_216 = _to_copy_118.view(262144, 256) ;  _to_copy_118 = None
        mm_44 = torch.mm(view_216,t_44) ;  view_216 = t_44 = None
        view_217 = mm_44.view(1, 512, 512, 512) ;  mm_44 = None
        sigmoid_16 = torch.sigmoid(view_217) ;  view_217 = None
        mul_27 = torch.mul(view_215,sigmoid_16) ;  view_215 = sigmoid_16 = None
        view_218 = mul_27.view(262144, 512) ;  mul_27 = None
        view_219 = view_218.view(1, 512, 512, 512) ;  view_218 = None
        transpose_3 = torch.transpose(bitwise_and_4,1,2) 
        unsqueeze_63 = torch.unsqueeze(transpose_3,3) ;  transpose_3 = None
        clone_15 = torch.clone(unsqueeze_63,memory_format = torch.contiguous_format) ;  unsqueeze_63 = None
        bitwise_not_13 = torch.bitwise_not(clone_15) ;  clone_15 = None
        masked_fill_13 = view_219.masked_fill(bitwise_not_13,0) ;  view_219 = bitwise_not_13 = None
        view_220 = masked_fill_13.view(262144, 512) ;  masked_fill_13 = None
        view_224 = view_220.view(1, 512, 512, 512) 
        split_tensor_42 = torch.split(view_224,256,dim = -1) ;  view_224 = None
        getitem_261 = split_tensor_42[0]
        unsqueeze_66 = torch.unsqueeze(getitem_261,4) ;  getitem_261 = None
        permute_121 = unsqueeze_66.permute(0, 2, 4, 3, 1) ;  unsqueeze_66 = None
        permute_122 = permute_121.permute(3, 1, 4, 0, 2) ;  permute_121 = None
        view_225 = permute_122.view(256, 512, 512) ;  permute_122 = None
        view_226 = view_220.view(1, 512, 512, 512) ;  view_220 = None
        split_tensor_43 = torch.split(view_226,256,dim = -1) ;  view_226 = None
        getitem_264 = split_tensor_43[1];  split_tensor_43 = None
        unsqueeze_67 = torch.unsqueeze(getitem_264,4) ;  getitem_264 = None
        permute_123 = unsqueeze_67.permute(0, 4, 2, 3, 1) ;  unsqueeze_67 = None
        permute_124 = permute_123.permute(3, 4, 0, 2, 1) ;  permute_123 = None
        view_227 = permute_124.view(256, 512, 512) ;  permute_124 = None
        bmm_13 = torch.bmm(view_225,view_227) ;  view_225 = view_227 = None
        view_228 = bmm_13.view(256, 512, 1, 1, 512) ;  bmm_13 = None
        permute_125 = view_228.permute(3, 1, 4, 0, 2) ;  view_228 = None
        view_229 = permute_125.view(1, 512, 512, 256) ;  permute_125 = None
        _to_copy_119 = view_213.to(dtype = torch.float32) ;  view_213 = None
        native_layer_norm_default_25 = (torch.nn.functional.layer_norm(_to_copy_119,[256],None,None,1e-05),) ;  _to_copy_119 = None
        getitem_265 = native_layer_norm_default_25[0]
        _to_copy_120 = view_229.to(dtype = torch.float32) ;  view_229 = None
        native_layer_norm_default_26 = (torch.nn.functional.layer_norm(_to_copy_120,[256],None,None,1e-05),) ;  _to_copy_120 = None
        getitem_268 = native_layer_norm_default_26[0]
        add_30 = torch.add(getitem_265,getitem_268) ;  getitem_265 = getitem_268 = None
        _to_copy_121 = arg83_1.to(dtype = torch.bfloat16) ;  arg83_1 = None
        _to_copy_122 = add_30.to(dtype = torch.bfloat16) ;  add_30 = None
        t_45 = _to_copy_121.t() ;  _to_copy_121 = None
        view_230 = _to_copy_122.view(262144, 256) ;  _to_copy_122 = None
        mm_45 = torch.mm(view_230,t_45) ;  view_230 = t_45 = None
        view_231 = mm_45.view(1, 512, 512, 256) ;  mm_45 = None
        _to_copy_123 = getitem_252.to(dtype = torch.bfloat16) ;  getitem_252 = None
        _to_copy_124 = getitem_245.to(dtype = torch.bfloat16) ;  getitem_245 = None
        t_46 = _to_copy_123.t() ;  _to_copy_123 = None
        view_232 = _to_copy_124.view(262144, 256) ;  _to_copy_124 = None
        mm_46 = torch.mm(view_232,t_46) ;  view_232 = t_46 = None
        view_233 = mm_46.view(1, 512, 512, 256) ;  mm_46 = None
        sigmoid_17 = torch.sigmoid(view_233) ;  view_233 = None
        mul_28 = torch.mul(view_231,sigmoid_17) ;  view_231 = sigmoid_17 = None
        add_31 = torch.add(add_25,mul_28) ;  mul_28 = None
        _to_copy_125 = add_25.to(dtype = torch.float32) 
        native_layer_norm_default_27 = (torch.nn.functional.layer_norm(_to_copy_125,[256],arg86_1,arg87_1,1e-05),) ;  _to_copy_125 = arg86_1 = arg87_1 = None
        getitem_271 = native_layer_norm_default_27[0]
        _to_copy_126 = arg88_1.to(dtype = torch.bfloat16) ;  arg88_1 = None
        _to_copy_127 = getitem_271.to(dtype = torch.bfloat16) ;  getitem_271 = None
        t_47 = _to_copy_126.t() ;  _to_copy_126 = None
        view_234 = _to_copy_127.view(262144, 256) ;  _to_copy_127 = None
        mm_47 = torch.mm(view_234,t_47) ;  view_234 = t_47 = None
        view_235 = mm_47.view(1, 512, 512, 2056) ;  mm_47 = None
        split_with_sizes_default_11 = torch.split(view_235,[2048,8],dim = -1) ;  view_235 = None
        getitem_274 = split_with_sizes_default_11[0]
        getitem_275 = split_with_sizes_default_11[1];  split_with_sizes_default_11 = None
        view_236 = getitem_275.view(1, 512, 512, 2, 4) ;  getitem_275 = None
        permute_126 = view_236.permute(0, 3, 4, 1, 2) ;  view_236 = None
        view_237 = permute_126.view(1, 2, 4, 1, 512, 512) ;  permute_126 = None
        view_238 = bitwise_and_4.view(1, 1, 1, 1, 512, 512) 
        bitwise_not_14 = torch.bitwise_not(view_238) ;  view_238 = None
        masked_fill_14 = view_237.masked_fill(bitwise_not_14,-10000) ;  view_237 = bitwise_not_14 = None
        view_239 = masked_fill_14.view(1, 2, 4, 512, 512) ;  masked_fill_14 = None
        view_240 = view_239.view(8, 1, 512, 512) ;  view_239 = None
        split_tensor_44 = torch.split(getitem_274,1024,dim = -1) ;  getitem_274 = None
        getitem_276 = split_tensor_44[0]
        getitem_277 = split_tensor_44[1];  split_tensor_44 = None
        permute_127 = getitem_277.permute(0, 2, 1, 3) ;  getitem_277 = None
        stack_3 = torch.stack([getitem_276,permute_127]) ;  getitem_276 = permute_127 = None
        view_241 = stack_3.view(2, 1, 512, 512, 4, 4, 64) ;  stack_3 = None
        permute_128 = view_241.permute(4, 1, 0, 5, 2, 3, 6) ;  view_241 = None
        clone_16 = torch.clone(permute_128,memory_format = torch.contiguous_format) ;  permute_128 = None
        _unsafe_view_12 = clone_16.view(4, 8, 512, 512, 64) ;  clone_16 = None
        unbind_int_9 = torch.unbind(_unsafe_view_12) ;  _unsafe_view_12 = None
        getitem_278 = unbind_int_9[0]
        getitem_279 = unbind_int_9[1]
        getitem_280 = unbind_int_9[2]
        getitem_281 = unbind_int_9[3];  unbind_int_9 = None
        split_tensor_45 = torch.split(getitem_278,4) ;  getitem_278 = None
        getitem_282 = split_tensor_45[0]
        getitem_283 = split_tensor_45[1];  split_tensor_45 = None
        split_tensor_46 = torch.split(getitem_279,4) ;  getitem_279 = None
        getitem_284 = split_tensor_46[0]
        getitem_285 = split_tensor_46[1];  split_tensor_46 = None
        split_tensor_47 = torch.split(getitem_280,4) ;  getitem_280 = None
        getitem_286 = split_tensor_47[0]
        getitem_287 = split_tensor_47[1];  split_tensor_47 = None
        split_tensor_48 = torch.split(view_240,4) ;  view_240 = None
        getitem_288 = split_tensor_48[0]
        getitem_289 = split_tensor_48[1];  split_tensor_48 = None
        expand_9 = getitem_288.expand(4, 512, 512, 512) ;  getitem_288 = None
        _scaled_dot_product_efficient_attention_default_9 = (torch.nn.functional.scaled_dot_product_attention(getitem_282,getitem_284,getitem_286,expand_9,False),) ;  getitem_282 = getitem_284 = getitem_286 = expand_9 = None
        getitem_290 = _scaled_dot_product_efficient_attention_default_9[0]
        expand_10 = getitem_289.expand(4, 512, 512, 512) ;  getitem_289 = None
        _scaled_dot_product_efficient_attention_default_10 = (torch.nn.functional.scaled_dot_product_attention(getitem_283,getitem_285,getitem_287,expand_10,False),) ;  getitem_283 = getitem_285 = getitem_287 = expand_10 = None
        getitem_294 = _scaled_dot_product_efficient_attention_default_10[0]
        cat_3 = torch.cat([getitem_290,getitem_294]) ;  getitem_290 = getitem_294 = None
        sigmoid_18 = torch.sigmoid(getitem_281) ;  getitem_281 = None
        mul_29 = torch.mul(cat_3,sigmoid_18) ;  cat_3 = sigmoid_18 = None
        view_242 = mul_29.view(1, 2, 4, 512, 512, 64) ;  mul_29 = None
        permute_129 = view_242.permute(0, 3, 4, 1, 2, 5) ;  view_242 = None
        clone_17 = torch.clone(permute_129,memory_format = torch.contiguous_format) ;  permute_129 = None
        _unsafe_view_13 = clone_17.view(1, 512, 512, 512) ;  clone_17 = None
        _to_copy_128 = arg89_1.to(dtype = torch.bfloat16) ;  arg89_1 = None
        t_48 = _to_copy_128.t() ;  _to_copy_128 = None
        view_243 = _unsafe_view_13.view(262144, 512) ;  _unsafe_view_13 = None
        mm_48 = torch.mm(view_243,t_48) ;  view_243 = t_48 = None
        view_244 = mm_48.view(1, 512, 512, 512) ;  mm_48 = None
        view_245 = view_244.view(1, 512, 512, 2, 4, 64) ;  view_244 = None
        permute_130 = view_245.permute(3, 0, 1, 2, 4, 5) ;  view_245 = None
        view_246 = permute_130.view(2, 1, 512, 512, 256) ;  permute_130 = None
        unbind_int_10 = torch.unbind(view_246) ;  view_246 = None
        getitem_298 = unbind_int_10[0]
        getitem_299 = unbind_int_10[1];  unbind_int_10 = None
        permute_131 = getitem_299.permute(0, 2, 1, 3) ;  getitem_299 = None
        permute_132 = permute_131.permute(0, 2, 1, 3) ;  permute_131 = None
        add_32 = torch.add(getitem_298,permute_132) ;  getitem_298 = permute_132 = None
        add_33 = torch.add(add_31,add_32) ;  add_31 = add_32 = None
        split_tensor_49 = torch.split(add_25,512,dim = -2) 
        getitem_300 = split_tensor_49[0];  split_tensor_49 = None
        _to_copy_129 = getitem_300.to(dtype = torch.float32) ;  getitem_300 = None
        native_layer_norm_default_28 = (torch.nn.functional.layer_norm(_to_copy_129,[256],arg77_1,arg78_1,1e-05),) ;  _to_copy_129 = arg77_1 = arg78_1 = None
        getitem_301 = native_layer_norm_default_28[0]
        _to_copy_130 = arg79_1.to(dtype = torch.bfloat16) ;  arg79_1 = None
        _to_copy_131 = getitem_301.to(dtype = torch.bfloat16) ;  getitem_301 = None
        t_49 = _to_copy_130.t() ;  _to_copy_130 = None
        view_247 = _to_copy_131.view(262144, 256) ;  _to_copy_131 = None
        mm_49 = torch.mm(view_247,t_49) ;  view_247 = t_49 = None
        view_248 = mm_49.view(1, 512, 512, 1024) ;  mm_49 = None
        split_tensor_50 = torch.split(view_248,512,dim = -1) ;  view_248 = None
        getitem_304 = split_tensor_50[0]
        getitem_305 = split_tensor_50[1];  split_tensor_50 = None
        silu_6 = torch.nn.functional.silu(getitem_304) ;  getitem_304 = None
        mul_30 = torch.mul(silu_6,getitem_305) ;  silu_6 = getitem_305 = None
        _to_copy_132 = arg80_1.to(dtype = torch.bfloat16) ;  arg80_1 = None
        t_50 = _to_copy_132.t() ;  _to_copy_132 = None
        view_250 = mul_30.view(262144, 512) ;  mul_30 = None
        mm_50 = torch.mm(view_250,t_50) ;  view_250 = t_50 = None
        view_251 = mm_50.view(1, 512, 512, 256) ;  mm_50 = None
        add_34 = torch.add(add_33,view_251) ;  add_33 = view_251 = None
        _to_copy_133 = add_29.to(dtype = torch.float32) 
        native_layer_norm_default_29 = (torch.nn.functional.layer_norm(_to_copy_133,[384],arg94_1,arg95_1,1e-05),) ;  _to_copy_133 = arg94_1 = arg95_1 = None
        getitem_306 = native_layer_norm_default_29[0]
        _to_copy_134 = add_25.to(dtype = torch.float32) ;  add_25 = None
        native_layer_norm_default_30 = (torch.nn.functional.layer_norm(_to_copy_134,[256],arg96_1,arg97_1,1e-05),) ;  _to_copy_134 = arg96_1 = arg97_1 = None
        getitem_309 = native_layer_norm_default_30[0]
        _to_copy_135 = arg98_1.to(dtype = torch.bfloat16) ;  arg98_1 = None
        _to_copy_136 = getitem_309.to(dtype = torch.bfloat16) ;  getitem_309 = None
        t_51 = _to_copy_135.t() ;  _to_copy_135 = None
        view_252 = _to_copy_136.view(262144, 256) ;  _to_copy_136 = None
        mm_51 = torch.mm(view_252,t_51) ;  view_252 = t_51 = None
        view_253 = mm_51.view(1, 512, 512, 16) ;  mm_51 = None
        permute_133 = view_253.permute(0, 3, 1, 2) ;  view_253 = None
        view_254 = bitwise_and_4.view(1, 1, 512, 512) ;  bitwise_and_4 = None
        bitwise_not_15 = torch.bitwise_not(view_254) ;  view_254 = None
        masked_fill_15 = permute_133.masked_fill(bitwise_not_15,-10000) ;  permute_133 = bitwise_not_15 = None
        _to_copy_137 = getitem_306.to(dtype = torch.bfloat16) ;  getitem_306 = None
        _to_copy_138 = arg100_1.to(dtype = torch.bfloat16) ;  arg100_1 = None
        unsqueeze_68 = torch.unsqueeze(_to_copy_137,3) ;  _to_copy_137 = None
        unsqueeze_69 = torch.unsqueeze(unsqueeze_68,4) ;  unsqueeze_68 = None
        unsqueeze_70 = torch.unsqueeze(unsqueeze_69,5) ;  unsqueeze_69 = None
        permute_134 = unsqueeze_70.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_70 = None
        unsqueeze_71 = torch.unsqueeze(_to_copy_138,4) ;  _to_copy_138 = None
        unsqueeze_72 = torch.unsqueeze(unsqueeze_71,5) ;  unsqueeze_71 = None
        permute_135 = unsqueeze_72.permute(1, 4, 2, 5, 3, 0) ;  unsqueeze_72 = None
        permute_136 = permute_134.permute(3, 5, 0, 1, 2, 4) ;  permute_134 = None
        view_255 = permute_136.view(1, 512, 384) ;  permute_136 = None
        permute_137 = permute_135.permute(5, 0, 1, 2, 4, 3) ;  permute_135 = None
        view_256 = permute_137.view(1, 384, 1536) ;  permute_137 = None
        bmm_14 = torch.bmm(view_255,view_256) ;  view_255 = view_256 = None
        view_257 = bmm_14.view(512, 1, 4, 1, 16, 24) ;  bmm_14 = None
        permute_138 = view_257.permute(2, 3, 4, 0, 5, 1) ;  view_257 = None
        view_258 = permute_138.view(4, 1, 16, 512, 24) ;  permute_138 = None
        unbind_int_11 = torch.unbind(view_258) ;  view_258 = None
        getitem_312 = unbind_int_11[0]
        getitem_313 = unbind_int_11[1]
        getitem_314 = unbind_int_11[2]
        getitem_315 = unbind_int_11[3];  unbind_int_11 = None
        view_259 = arg99_1.view(1, 16, 1, 24) ;  arg99_1 = None
        add_35 = torch.add(getitem_312,view_259) ;  getitem_312 = view_259 = None
        _to_copy_139 = add_35.to(dtype = torch.bfloat16) ;  add_35 = None
        expand_11 = masked_fill_15.expand(1, 16, 512, 512) ;  masked_fill_15 = None
        _scaled_dot_product_efficient_attention_default_11 = (torch.nn.functional.scaled_dot_product_attention(_to_copy_139,getitem_313,getitem_314,expand_11,False),) ;  _to_copy_139 = getitem_313 = getitem_314 = expand_11 = None
        getitem_316 = _scaled_dot_product_efficient_attention_default_11[0]
        add_36 = torch.add(getitem_315,1) ;  getitem_315 = None
        sigmoid_19 = torch.sigmoid(add_36) ;  add_36 = None
        mul_31 = torch.mul(getitem_316,sigmoid_19) ;  getitem_316 = sigmoid_19 = None
        _to_copy_140 = arg101_1.to(dtype = torch.bfloat16) ;  arg101_1 = None
        unsqueeze_73 = torch.unsqueeze(mul_31,4) ;  mul_31 = None
        permute_139 = unsqueeze_73.permute(0, 2, 4, 3, 1) ;  unsqueeze_73 = None
        unsqueeze_74 = torch.unsqueeze(_to_copy_140,3) ;  _to_copy_140 = None
        unsqueeze_75 = torch.unsqueeze(unsqueeze_74,4) ;  unsqueeze_74 = None
        permute_140 = unsqueeze_75.permute(3, 4, 2, 1, 0) ;  unsqueeze_75 = None
        permute_141 = permute_139.permute(1, 3, 4, 0, 2) ;  permute_139 = None
        clone_18 = torch.clone(permute_141,memory_format = torch.contiguous_format) ;  permute_141 = None
        _unsafe_view_14 = clone_18.view(1, 512, 384) ;  clone_18 = None
        permute_142 = permute_140.permute(3, 4, 0, 2, 1) ;  permute_140 = None
        clone_19 = torch.clone(permute_142,memory_format = torch.contiguous_format) ;  permute_142 = None
        _unsafe_view_15 = clone_19.view(1, 384, 384) ;  clone_19 = None
        bmm_15 = torch.bmm(_unsafe_view_14,_unsafe_view_15) ;  _unsafe_view_14 = _unsafe_view_15 = None
        view_260 = bmm_15.view(512, 1, 1, 1, 384) ;  bmm_15 = None
        permute_143 = view_260.permute(3, 0, 4, 1, 2) ;  view_260 = None
        view_261 = permute_143.view(1, 512, 384) ;  permute_143 = None
        unsqueeze_76 = torch.unsqueeze(arg109_1,-1) ;  arg109_1 = None
        mul_32 = torch.mul(view_261,unsqueeze_76) ;  view_261 = unsqueeze_76 = None
        add_37 = torch.add(add_29,mul_32) ;  mul_32 = None
        split_tensor_51 = torch.split(add_29,512,dim = -2) ;  add_29 = None
        getitem_320 = split_tensor_51[0];  split_tensor_51 = None
        _to_copy_141 = getitem_320.to(dtype = torch.float32) ;  getitem_320 = None
        native_layer_norm_default_31 = (torch.nn.functional.layer_norm(_to_copy_141,[384],arg90_1,arg91_1,1e-05),) ;  _to_copy_141 = arg90_1 = arg91_1 = None
        getitem_321 = native_layer_norm_default_31[0]
        _to_copy_142 = arg92_1.to(dtype = torch.bfloat16) ;  arg92_1 = None
        _to_copy_143 = getitem_321.to(dtype = torch.bfloat16) ;  getitem_321 = None
        t_52 = _to_copy_142.t() ;  _to_copy_142 = None
        view_262 = _to_copy_143.view(512, 384) ;  _to_copy_143 = None
        mm_52 = torch.mm(view_262,t_52) ;  view_262 = t_52 = None
        view_263 = mm_52.view(1, 512, 1536) ;  mm_52 = None
        split_tensor_52 = torch.split(view_263,768,dim = -1) ;  view_263 = None
        getitem_324 = split_tensor_52[0]
        getitem_325 = split_tensor_52[1];  split_tensor_52 = None
        silu_7 = torch.nn.functional.silu(getitem_324) ;  getitem_324 = None
        mul_33 = torch.mul(silu_7,getitem_325) ;  silu_7 = getitem_325 = None
        _to_copy_144 = arg93_1.to(dtype = torch.bfloat16) ;  arg93_1 = None
        t_53 = _to_copy_144.t() ;  _to_copy_144 = None
        view_265 = mul_33.view(512, 768) ;  mul_33 = None
        mm_53 = torch.mm(view_265,t_53) ;  view_265 = t_53 = None
        view_266 = mm_53.view(1, 512, 384) ;  mm_53 = None
        add_38 = torch.add(add_37,view_266) ;  add_37 = view_266 = None
        _to_copy_145 = add_38.to(dtype = torch.float32) ;  add_38 = None
        native_layer_norm_default_32 = (torch.nn.functional.layer_norm(_to_copy_145,[384],None,None,1e-05),) ;  _to_copy_145 = None
        getitem_326 = native_layer_norm_default_32[0]
        _to_copy_146 = add_34.to(dtype = torch.float32) ;  add_34 = None
        native_layer_norm_default_33 = (torch.nn.functional.layer_norm(_to_copy_146,[256],None,None,1e-05),) ;  _to_copy_146 = None
        getitem_329 = native_layer_norm_default_33[0]
        _to_copy_147 = arg103_1.to(dtype = torch.bfloat16) ;  arg103_1 = None
        _to_copy_148 = getitem_329.to(dtype = torch.bfloat16) 
        t_54 = _to_copy_147.t() ;  _to_copy_147 = None
        view_267 = _to_copy_148.view(262144, 256) ;  _to_copy_148 = None
        mm_54 = torch.mm(view_267,t_54) ;  view_267 = t_54 = None
        view_268 = mm_54.view(1, 512, 512, 64) ;  mm_54 = None
        permute_144 = getitem_329.permute(0, 2, 1, 3) 
        add_39 = torch.add(getitem_329,permute_144) ;  getitem_329 = permute_144 = None
        _to_copy_149 = arg104_1.to(dtype = torch.bfloat16) ;  arg104_1 = None
        _to_copy_150 = add_39.to(dtype = torch.bfloat16) ;  add_39 = None
        t_55 = _to_copy_149.t() ;  _to_copy_149 = None
        view_269 = _to_copy_150.view(262144, 256) ;  _to_copy_150 = None
        mm_55 = torch.mm(view_269,t_55) ;  view_269 = t_55 = None
        view_270 = mm_55.view(1, 512, 512, 64) ;  mm_55 = None
        _to_copy_151 = arg102_1.to(dtype = torch.bfloat16) ;  arg102_1 = None
        _to_copy_152 = getitem_326.to(dtype = torch.bfloat16) ;  getitem_326 = None
        t_56 = _to_copy_151.t() ;  _to_copy_151 = None
        view_271 = _to_copy_152.view(512, 384) ;  _to_copy_152 = None
        mm_56 = torch.mm(view_271,t_56) ;  view_271 = t_56 = None
        view_272 = mm_56.view(1, 512, 1850) ;  mm_56 = None
        view_273 = view_272.view(1, 512, 37, 50) ;  view_272 = None
        arange_2 = torch.arange(1,device = self.device,pin_memory = False) 
        unsqueeze_77 = torch.unsqueeze(arange_2,1) ;  arange_2 = None
        index_2 = view_273[unsqueeze_77,arg113_1,arg114_1] ;  view_273 = unsqueeze_77 = arg113_1 = arg114_1 = None
        return (view_268, view_270, index_2)
        
    # To see more debug info, please use `graph_module.print_readable()`