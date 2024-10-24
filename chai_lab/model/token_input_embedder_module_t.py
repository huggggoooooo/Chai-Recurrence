import torch
import torch.nn as nn

class token_input_embedder_module(nn.Module):
    def  __init__(self, model):
        super(token_input_embedder_module, self).__init__()
        self.device = torch.device('cuda')
        self.arg0_1 = model.token_single_input_emb.atom_encoder.to_atom_cond.weight
        self.arg1_1 = getattr(model.token_single_input_emb.atom_encoder.pair_update_block.atom_single_to_atom_pair_proj_h, "1").weight
        self.arg2_1 = getattr(model.token_single_input_emb.atom_encoder.pair_update_block.atom_single_to_atom_pair_proj_w, "1").weight
        self.arg3_1 = getattr(model.token_single_input_emb.atom_encoder.pair_update_block.atom_pair_mlp, "0").weight
        self.arg4_1 = getattr(model.token_single_input_emb.atom_encoder.pair_update_block.atom_pair_mlp, "2").weight
        self.arg5_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "0").ada_ln.lin_s_merged.weight
        self.arg6_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_a_nobias_double.weight
        self.arg7_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_b_nobias.weight
        self.arg8_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.weight
        self.arg9_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.bias
        self.arg10_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "1").ada_ln.lin_s_merged.weight
        self.arg11_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_a_nobias_double.weight
        self.arg12_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_b_nobias.weight
        self.arg13_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.weight
        self.arg14_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.bias
        self.arg15_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "2").ada_ln.lin_s_merged.weight
        self.arg16_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_a_nobias_double.weight
        self.arg17_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_b_nobias.weight
        self.arg18_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.weight
        self.arg19_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.bias
        self.arg20_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").q_bias
        self.arg21_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").single_layer_norm.lin_s_merged.weight
        self.arg22_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").to_qkv.weight
        self.arg23_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.weight
        self.arg24_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.bias
        self.arg25_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").q_bias
        self.arg26_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").single_layer_norm.lin_s_merged.weight
        self.arg27_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").to_qkv.weight
        self.arg28_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.weight
        self.arg29_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.bias
        self.arg30_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").q_bias
        self.arg31_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").single_layer_norm.lin_s_merged.weight
        self.arg32_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").to_qkv.weight
        self.arg33_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.weight
        self.arg34_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.bias
        self.arg35_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").weight
        self.arg36_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").bias
        self.arg37_1 = getattr(model.token_single_input_emb.atom_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "1").weight
        self.arg38_1 = getattr(model.token_single_input_emb.atom_encoder.to_token_single, "0").weight
        self.arg39_1 = model.token_single_proj_in_trunk.weight
        self.arg40_1 = model.token_single_proj_in_structure.weight
        self.arg41_1 = model.token_pair_proj_in_trunk.weight
        self.arg42_1 = model.token_single_to_token_pair_outer_sum_proj.weight

    def forward(self, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1):
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

        _to_copy = arg0_1.to(dtype = torch.bfloat16) ;  arg0_1 = None
        t = _to_copy.t() ;  _to_copy = None
        view = arg45_1.view(11776, 128) ;  arg45_1 = None
        mm = torch.mm(view,t) ;  view = t = None
        view_1 = mm.view(1, 11776, 128) ;  mm = None
        unsqueeze = torch.unsqueeze(view_1,1) 
        expand = unsqueeze.expand(-1, 1, -1, -1) ;  unsqueeze = None
        _to_copy_1 = view_1.to(dtype = torch.float32) ;  view_1 = None
        native_layer_norm_default = (torch.nn.functional.layer_norm(_to_copy_1,[128],None,None,1e-05),) ;  _to_copy_1 = None
        getitem = native_layer_norm_default[0]
        unsqueeze_1 = torch.unsqueeze(getitem,1) 
        expand_1 = unsqueeze_1.expand(-1, 1, -1, -1) ;  unsqueeze_1 = None
        slice_1 = expand_1[0:] 
        slice_2 = slice_1[:, 0:] ;  slice_1 = None
        index = slice_2[:,:,arg48_1] ;  slice_2 = arg48_1 = None
        slice_3 = expand_1[0:] ;  expand_1 = None
        slice_4 = slice_3[:, 0:] ;  slice_3 = None
        index_1 = slice_4[:,:,arg49_1] ;  slice_4 = arg49_1 = None
        relu = torch.relu(index) ;  index = None
        _to_copy_2 = arg1_1.to(dtype = torch.bfloat16) ;  arg1_1 = None
        _to_copy_3 = relu.to(dtype = torch.bfloat16) ;  relu = None
        t_1 = _to_copy_2.t() ;  _to_copy_2 = None
        view_2 = _to_copy_3.view(11776, 128) ;  _to_copy_3 = None
        mm_1 = torch.mm(view_2,t_1) ;  view_2 = t_1 = None
        view_3 = mm_1.view(1, 1, 368, 32, 16) ;  mm_1 = None
        relu_1 = torch.relu(index_1) ;  index_1 = None
        _to_copy_4 = arg2_1.to(dtype = torch.bfloat16) ;  arg2_1 = None
        _to_copy_5 = relu_1.to(dtype = torch.bfloat16) ;  relu_1 = None
        t_2 = _to_copy_4.t() ;  _to_copy_4 = None
        view_4 = _to_copy_5.view(47104, 128) ;  _to_copy_5 = None
        mm_2 = torch.mm(view_4,t_2) ;  view_4 = t_2 = None
        view_5 = mm_2.view(1, 1, 368, 128, 16) ;  mm_2 = None
        unsqueeze_2 = torch.unsqueeze(arg46_1,1) ;  arg46_1 = None
        view_6 = view_3.view(1, 1, 368, 32, 1, 16) ;  view_3 = None
        add = torch.add(unsqueeze_2,view_6) ;  unsqueeze_2 = view_6 = None
        view_7 = view_5.view(1, 1, 368, 1, 128, 16) ;  view_5 = None
        add_1 = torch.add(add,view_7) ;  add = view_7 = None
        _to_copy_6 = arg3_1.to(dtype = torch.bfloat16) ;  arg3_1 = None
        t_3 = _to_copy_6.t() ;  _to_copy_6 = None
        view_8 = add_1.view(1507328, 16) 
        mm_3 = torch.mm(view_8,t_3) ;  view_8 = t_3 = None
        view_9 = mm_3.view(1, 1, 368, 32, 128, 16) ;  mm_3 = None
        relu_2 = torch.relu(view_9) ;  view_9 = None
        view_10 = relu_2.view(1507328, 16) ;  relu_2 = None
        _to_copy_7 = arg4_1.to(dtype = torch.bfloat16) ;  arg4_1 = None
        t_4 = _to_copy_7.t() ;  _to_copy_7 = None
        view_13 = view_10.view(1, 1, 368, 32, 128, 16) ;  view_10 = None
        view_14 = view_13.view(1507328, 16) ;  view_13 = None
        mm_4 = torch.mm(view_14,t_4) ;  view_14 = t_4 = None
        view_15 = mm_4.view(1, 1, 368, 32, 128, 16) ;  mm_4 = None
        add_2 = torch.add(view_15,add_1) ;  view_15 = add_1 = None
        view_16 = add_2.view(1507328, 16) ;  add_2 = None
        view_18 = expand.view(1, 11776, 128) ;  expand = None
        unsqueeze_3 = torch.unsqueeze(getitem,1) ;  getitem = None
        expand_2 = unsqueeze_3.expand(-1, 1, -1, -1) ;  unsqueeze_3 = None
        view_19 = expand_2.view(1, 11776, 128) ;  expand_2 = None
        unsqueeze_4 = torch.unsqueeze(arg47_1,1) ;  arg47_1 = None
        expand_3 = unsqueeze_4.expand(-1, 1, -1, -1, -1) ;  unsqueeze_4 = None
        view_21 = expand_3.view(1, 368, 32, 128) ;  expand_3 = None
        unsqueeze_5 = torch.unsqueeze(arg50_1,1) 
        expand_4 = unsqueeze_5.expand(-1, 1, -1) ;  unsqueeze_5 = None
        view_22 = expand_4.view(1, 11776) ;  expand_4 = None
        arange = torch.arange(11776,device = self.device,pin_memory = False) 
        view_23 = arange.view(368, 32) ;  arange = None
        slice_5 = view_23[0:] ;  view_23 = None
        slice_6 = slice_5[:, 0:1] ;  slice_5 = None
        add_3 = torch.add(slice_6,-48) ;  slice_6 = None
        arange_1 = torch.arange(128,device = self.device,pin_memory = False) 
        add_4 = torch.add(add_3,arange_1) ;  add_3 = arange_1 = None
        remainder = torch.remainder(add_4,11776) ;  add_4 = None
        view_24 = view_16.view(1, 1, 368, 32, 128, 16) ;  view_16 = None
        view_25 = view_24.view(1, 368, 32, 128, 16) ;  view_24 = None
        _to_copy_8 = view_25.to(dtype = torch.float32) ;  view_25 = None
        native_layer_norm_default_1 = (torch.nn.functional.layer_norm(_to_copy_8,[16],arg35_1,arg36_1,1e-05),) ;  _to_copy_8 = arg35_1 = arg36_1 = None
        getitem_3 = native_layer_norm_default_1[0]
        unbind_int = torch.unbind(arg37_1) ;  arg37_1 = None
        getitem_6 = unbind_int[0]
        getitem_7 = unbind_int[1]
        getitem_8 = unbind_int[2];  unbind_int = None
        unsqueeze_6 = torch.unsqueeze(view_22,-1) 
        bitwise_not = torch.bitwise_not(unsqueeze_6) ;  unsqueeze_6 = None
        masked_fill = view_18.masked_fill(bitwise_not,0.0) ;  view_18 = bitwise_not = None
        _to_copy_9 = getitem_3.to(dtype = torch.bfloat16) 
        _to_copy_10 = getitem_6.to(dtype = torch.bfloat16) ;  getitem_6 = None
        unsqueeze_7 = torch.unsqueeze(_to_copy_9,5) ;  _to_copy_9 = None
        permute = unsqueeze_7.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_7 = None
        unsqueeze_8 = torch.unsqueeze(_to_copy_10,2) ;  _to_copy_10 = None
        unsqueeze_9 = torch.unsqueeze(unsqueeze_8,3) ;  unsqueeze_8 = None
        unsqueeze_10 = torch.unsqueeze(unsqueeze_9,4) ;  unsqueeze_9 = None
        unsqueeze_11 = torch.unsqueeze(unsqueeze_10,5) ;  unsqueeze_10 = None
        permute_1 = unsqueeze_11.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_11 = None
        permute_2 = permute.permute(2, 3, 4, 5, 0, 1) ;  permute = None
        view_26 = permute_2.view(1, 1507328, 16) ;  permute_2 = None
        permute_3 = permute_1.permute(5, 0, 1, 2, 3, 4) ;  permute_1 = None
        view_27 = permute_3.view(1, 16, 4) ;  permute_3 = None
        bmm = torch.bmm(view_26,view_27) ;  view_26 = view_27 = None
        view_28 = bmm.view(368, 32, 128, 1, 1, 4) ;  bmm = None
        permute_4 = view_28.permute(4, 5, 0, 1, 2, 3) ;  view_28 = None
        view_29 = permute_4.view(1, 4, 368, 32, 128) ;  permute_4 = None
        view_30 = view_21.view(1, 1, 368, 32, 128) 
        bitwise_not_1 = torch.bitwise_not(view_30) ;  view_30 = None
        masked_fill_1 = view_29.masked_fill(bitwise_not_1,-10000) ;  view_29 = bitwise_not_1 = None
        _to_copy_11 = masked_fill.to(dtype = torch.float32) 
        native_layer_norm_default_2 = (torch.nn.functional.layer_norm(_to_copy_11,[128],None,None,0.1),) ;  _to_copy_11 = None
        getitem_9 = native_layer_norm_default_2[0]
        _to_copy_12 = arg21_1.to(dtype = torch.bfloat16) ;  arg21_1 = None
        _to_copy_13 = view_19.to(dtype = torch.bfloat16) 
        t_5 = _to_copy_12.t() ;  _to_copy_12 = None
        view_31 = _to_copy_13.view(11776, 128) ;  _to_copy_13 = None
        mm_5 = torch.mm(view_31,t_5) ;  view_31 = t_5 = None
        view_32 = mm_5.view(1, 11776, 256) ;  mm_5 = None
        split_tensor = torch.split(view_32,128,dim = -1) ;  view_32 = None
        getitem_12 = split_tensor[0]
        getitem_13 = split_tensor[1];  split_tensor = None
        add_5 = torch.add(getitem_12,1) ;  getitem_12 = None
        mul = torch.mul(getitem_9,add_5) ;  getitem_9 = add_5 = None
        add_6 = torch.add(mul,getitem_13) ;  mul = getitem_13 = None
        _to_copy_14 = add_6.to(dtype = torch.bfloat16) ;  add_6 = None
        _to_copy_15 = arg22_1.to(dtype = torch.bfloat16) ;  arg22_1 = None
        unsqueeze_12 = torch.unsqueeze(_to_copy_14,3) ;  _to_copy_14 = None
        unsqueeze_13 = torch.unsqueeze(unsqueeze_12,4) ;  unsqueeze_12 = None
        unsqueeze_14 = torch.unsqueeze(unsqueeze_13,5) ;  unsqueeze_13 = None
        permute_5 = unsqueeze_14.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_14 = None
        unsqueeze_15 = torch.unsqueeze(_to_copy_15,4) ;  _to_copy_15 = None
        unsqueeze_16 = torch.unsqueeze(unsqueeze_15,5) ;  unsqueeze_15 = None
        permute_6 = unsqueeze_16.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_16 = None
        permute_7 = permute_5.permute(3, 5, 0, 1, 2, 4) ;  permute_5 = None
        view_33 = permute_7.view(1, 11776, 128) ;  permute_7 = None
        permute_8 = permute_6.permute(5, 0, 1, 2, 4, 3) ;  permute_6 = None
        view_34 = permute_8.view(1, 128, 384) ;  permute_8 = None
        bmm_1 = torch.bmm(view_33,view_34) ;  view_33 = view_34 = None
        view_35 = bmm_1.view(11776, 1, 3, 1, 4, 32) ;  bmm_1 = None
        permute_9 = view_35.permute(2, 3, 4, 0, 5, 1) ;  view_35 = None
        view_36 = permute_9.view(3, 1, 4, 11776, 32) ;  permute_9 = None
        view_37 = view_36.view(3, 4, 11776, 32) ;  view_36 = None
        unbind_int_1 = torch.unbind(view_37) ;  view_37 = None
        getitem_14 = unbind_int_1[0]
        getitem_15 = unbind_int_1[1]
        getitem_16 = unbind_int_1[2];  unbind_int_1 = None
        unsqueeze_17 = torch.unsqueeze(arg20_1,0) ;  arg20_1 = None
        expand_5 = unsqueeze_17.expand(1, -1, -1) ;  unsqueeze_17 = None
        view_38 = expand_5.view(4, 1, 32) ;  expand_5 = None
        add_7 = torch.add(getitem_14,view_38) ;  getitem_14 = view_38 = None
        view_39 = add_7.view(4, 368, 32, 32) ;  add_7 = None
        slice_7 = getitem_15[0:] ;  getitem_15 = None
        slice_8 = slice_7[:, :, 0:] ;  slice_7 = None
        index_2 = slice_8[:,remainder] ;  slice_8 = None
        slice_9 = getitem_16[0:] ;  getitem_16 = None
        slice_10 = slice_9[:, :, 0:] ;  slice_9 = None
        index_3 = slice_10[:,remainder] ;  slice_10 = None
        view_40 = masked_fill_1.view(4, 368, 32, 128) ;  masked_fill_1 = None
        _to_copy_16 = view_39.to(dtype = torch.bfloat16) ;  view_39 = None
        expand_6 = view_40.expand(4, 368, 32, 128) ;  view_40 = None
        _scaled_dot_product_efficient_attention_default = (torch.nn.functional.scaled_dot_product_attention(_to_copy_16,index_2,index_3,expand_6,False),) ;  _to_copy_16 = index_2 = index_3 = expand_6 = None
        getitem_17 = _scaled_dot_product_efficient_attention_default[0]
        view_41 = getitem_17.view(1, 4, 368, 32, 32) ;  getitem_17 = None
        permute_10 = view_41.permute(0, 2, 3, 1, 4) ;  view_41 = None
        clone = torch.clone(permute_10,memory_format = torch.contiguous_format) ;  permute_10 = None
        _unsafe_view = clone.view(1, 11776, 128) ;  clone = None
        _to_copy_17 = arg24_1.to(dtype = torch.bfloat16) ;  arg24_1 = None
        _to_copy_18 = arg23_1.to(dtype = torch.bfloat16) ;  arg23_1 = None
        _to_copy_19 = view_19.to(dtype = torch.bfloat16) 
        view_42 = _to_copy_19.view(11776, 128) ;  _to_copy_19 = None
        t_6 = _to_copy_18.t() ;  _to_copy_18 = None
        addmm = torch.addmm(_to_copy_17,view_42,t_6) ;  _to_copy_17 = view_42 = t_6 = None
        view_43 = addmm.view(1, 11776, 128) ;  addmm = None
        sigmoid = torch.sigmoid(view_43) ;  view_43 = None
        view_44 = sigmoid.view(11776, 128) ;  sigmoid = None
        view_45 = view_44.view(1, 11776, 128) ;  view_44 = None
        mul_1 = torch.mul(_unsafe_view,view_45) ;  _unsafe_view = view_45 = None
        _to_copy_20 = masked_fill.to(dtype = torch.float32) 
        native_layer_norm_default_3 = (torch.nn.functional.layer_norm(_to_copy_20,[128],None,None,0.1),) ;  _to_copy_20 = None
        getitem_21 = native_layer_norm_default_3[0]
        _to_copy_21 = arg5_1.to(dtype = torch.bfloat16) ;  arg5_1 = None
        _to_copy_22 = view_19.to(dtype = torch.bfloat16) 
        t_7 = _to_copy_21.t() ;  _to_copy_21 = None
        view_46 = _to_copy_22.view(11776, 128) ;  _to_copy_22 = None
        mm_6 = torch.mm(view_46,t_7) ;  view_46 = t_7 = None
        view_47 = mm_6.view(1, 11776, 256) ;  mm_6 = None
        split_tensor_1 = torch.split(view_47,128,dim = -1) ;  view_47 = None
        getitem_24 = split_tensor_1[0]
        getitem_25 = split_tensor_1[1];  split_tensor_1 = None
        add_8 = torch.add(getitem_24,1) ;  getitem_24 = None
        mul_2 = torch.mul(getitem_21,add_8) ;  getitem_21 = add_8 = None
        add_9 = torch.add(mul_2,getitem_25) ;  mul_2 = getitem_25 = None
        _to_copy_23 = arg6_1.to(dtype = torch.bfloat16) ;  arg6_1 = None
        _to_copy_24 = add_9.to(dtype = torch.bfloat16) ;  add_9 = None
        t_8 = _to_copy_23.t() ;  _to_copy_23 = None
        view_48 = _to_copy_24.view(11776, 128) ;  _to_copy_24 = None
        mm_7 = torch.mm(view_48,t_8) ;  view_48 = t_8 = None
        view_49 = mm_7.view(1, 11776, 512) ;  mm_7 = None
        split_tensor_2 = torch.split(view_49,256,dim = -1) ;  view_49 = None
        getitem_26 = split_tensor_2[0]
        getitem_27 = split_tensor_2[1];  split_tensor_2 = None
        silu = torch.nn.functional.silu(getitem_26) ;  getitem_26 = None
        mul_3 = torch.mul(silu,getitem_27) ;  silu = getitem_27 = None
        _to_copy_25 = arg9_1.to(dtype = torch.bfloat16) ;  arg9_1 = None
        _to_copy_26 = arg8_1.to(dtype = torch.bfloat16) ;  arg8_1 = None
        _to_copy_27 = view_19.to(dtype = torch.bfloat16) 
        view_50 = _to_copy_27.view(11776, 128) ;  _to_copy_27 = None
        t_9 = _to_copy_26.t() ;  _to_copy_26 = None
        addmm_1 = torch.addmm(_to_copy_25,view_50,t_9) ;  _to_copy_25 = view_50 = t_9 = None
        view_51 = addmm_1.view(1, 11776, 128) ;  addmm_1 = None
        sigmoid_1 = torch.sigmoid(view_51) ;  view_51 = None
        _to_copy_28 = arg7_1.to(dtype = torch.bfloat16) ;  arg7_1 = None
        t_10 = _to_copy_28.t() ;  _to_copy_28 = None
        view_52 = mul_3.view(11776, 256) ;  mul_3 = None
        mm_8 = torch.mm(view_52,t_10) ;  view_52 = t_10 = None
        view_53 = mm_8.view(1, 11776, 128) ;  mm_8 = None
        mul_4 = torch.mul(sigmoid_1,view_53) ;  sigmoid_1 = view_53 = None
        add_10 = torch.add(masked_fill,mul_4) ;  masked_fill = mul_4 = None
        add_11 = torch.add(add_10,mul_1) ;  add_10 = mul_1 = None
        unsqueeze_18 = torch.unsqueeze(view_22,-1) 
        bitwise_not_2 = torch.bitwise_not(unsqueeze_18) ;  unsqueeze_18 = None
        masked_fill_2 = add_11.masked_fill(bitwise_not_2,0.0) ;  add_11 = bitwise_not_2 = None
        _to_copy_29 = getitem_3.to(dtype = torch.bfloat16) 
        _to_copy_30 = getitem_7.to(dtype = torch.bfloat16) ;  getitem_7 = None
        unsqueeze_19 = torch.unsqueeze(_to_copy_29,5) ;  _to_copy_29 = None
        permute_11 = unsqueeze_19.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_19 = None
        unsqueeze_20 = torch.unsqueeze(_to_copy_30,2) ;  _to_copy_30 = None
        unsqueeze_21 = torch.unsqueeze(unsqueeze_20,3) ;  unsqueeze_20 = None
        unsqueeze_22 = torch.unsqueeze(unsqueeze_21,4) ;  unsqueeze_21 = None
        unsqueeze_23 = torch.unsqueeze(unsqueeze_22,5) ;  unsqueeze_22 = None
        permute_12 = unsqueeze_23.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_23 = None
        permute_13 = permute_11.permute(2, 3, 4, 5, 0, 1) ;  permute_11 = None
        view_54 = permute_13.view(1, 1507328, 16) ;  permute_13 = None
        permute_14 = permute_12.permute(5, 0, 1, 2, 3, 4) ;  permute_12 = None
        view_55 = permute_14.view(1, 16, 4) ;  permute_14 = None
        bmm_2 = torch.bmm(view_54,view_55) ;  view_54 = view_55 = None
        view_56 = bmm_2.view(368, 32, 128, 1, 1, 4) ;  bmm_2 = None
        permute_15 = view_56.permute(4, 5, 0, 1, 2, 3) ;  view_56 = None
        view_57 = permute_15.view(1, 4, 368, 32, 128) ;  permute_15 = None
        view_58 = view_21.view(1, 1, 368, 32, 128) 
        bitwise_not_3 = torch.bitwise_not(view_58) ;  view_58 = None
        masked_fill_3 = view_57.masked_fill(bitwise_not_3,-10000) ;  view_57 = bitwise_not_3 = None
        _to_copy_31 = masked_fill_2.to(dtype = torch.float32) 
        native_layer_norm_default_4 = (torch.nn.functional.layer_norm(_to_copy_31,[128],None,None,0.1),) ;  _to_copy_31 = None
        getitem_28 = native_layer_norm_default_4[0]
        _to_copy_32 = arg26_1.to(dtype = torch.bfloat16) ;  arg26_1 = None
        _to_copy_33 = view_19.to(dtype = torch.bfloat16) 
        t_11 = _to_copy_32.t() ;  _to_copy_32 = None
        view_59 = _to_copy_33.view(11776, 128) ;  _to_copy_33 = None
        mm_9 = torch.mm(view_59,t_11) ;  view_59 = t_11 = None
        view_60 = mm_9.view(1, 11776, 256) ;  mm_9 = None
        split_tensor_3 = torch.split(view_60,128,dim = -1) ;  view_60 = None
        getitem_31 = split_tensor_3[0]
        getitem_32 = split_tensor_3[1];  split_tensor_3 = None
        add_12 = torch.add(getitem_31,1) ;  getitem_31 = None
        mul_5 = torch.mul(getitem_28,add_12) ;  getitem_28 = add_12 = None
        add_13 = torch.add(mul_5,getitem_32) ;  mul_5 = getitem_32 = None
        _to_copy_34 = add_13.to(dtype = torch.bfloat16) ;  add_13 = None
        _to_copy_35 = arg27_1.to(dtype = torch.bfloat16) ;  arg27_1 = None
        unsqueeze_24 = torch.unsqueeze(_to_copy_34,3) ;  _to_copy_34 = None
        unsqueeze_25 = torch.unsqueeze(unsqueeze_24,4) ;  unsqueeze_24 = None
        unsqueeze_26 = torch.unsqueeze(unsqueeze_25,5) ;  unsqueeze_25 = None
        permute_16 = unsqueeze_26.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_26 = None
        unsqueeze_27 = torch.unsqueeze(_to_copy_35,4) ;  _to_copy_35 = None
        unsqueeze_28 = torch.unsqueeze(unsqueeze_27,5) ;  unsqueeze_27 = None
        permute_17 = unsqueeze_28.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_28 = None
        permute_18 = permute_16.permute(3, 5, 0, 1, 2, 4) ;  permute_16 = None
        view_61 = permute_18.view(1, 11776, 128) ;  permute_18 = None
        permute_19 = permute_17.permute(5, 0, 1, 2, 4, 3) ;  permute_17 = None
        view_62 = permute_19.view(1, 128, 384) ;  permute_19 = None
        bmm_3 = torch.bmm(view_61,view_62) ;  view_61 = view_62 = None
        view_63 = bmm_3.view(11776, 1, 3, 1, 4, 32) ;  bmm_3 = None
        permute_20 = view_63.permute(2, 3, 4, 0, 5, 1) ;  view_63 = None
        view_64 = permute_20.view(3, 1, 4, 11776, 32) ;  permute_20 = None
        view_65 = view_64.view(3, 4, 11776, 32) ;  view_64 = None
        unbind_int_2 = torch.unbind(view_65) ;  view_65 = None
        getitem_33 = unbind_int_2[0]
        getitem_34 = unbind_int_2[1]
        getitem_35 = unbind_int_2[2];  unbind_int_2 = None
        unsqueeze_29 = torch.unsqueeze(arg25_1,0) ;  arg25_1 = None
        expand_7 = unsqueeze_29.expand(1, -1, -1) ;  unsqueeze_29 = None
        view_66 = expand_7.view(4, 1, 32) ;  expand_7 = None
        add_14 = torch.add(getitem_33,view_66) ;  getitem_33 = view_66 = None
        view_67 = add_14.view(4, 368, 32, 32) ;  add_14 = None
        slice_11 = getitem_34[0:] ;  getitem_34 = None
        slice_12 = slice_11[:, :, 0:] ;  slice_11 = None
        index_4 = slice_12[:,remainder] ;  slice_12 = None
        slice_13 = getitem_35[0:] ;  getitem_35 = None
        slice_14 = slice_13[:, :, 0:] ;  slice_13 = None
        index_5 = slice_14[:,remainder] ;  slice_14 = None
        view_68 = masked_fill_3.view(4, 368, 32, 128) ;  masked_fill_3 = None
        _to_copy_36 = view_67.to(dtype = torch.bfloat16) ;  view_67 = None
        expand_8 = view_68.expand(4, 368, 32, 128) ;  view_68 = None
        _scaled_dot_product_efficient_attention_default_1 = (torch.nn.functional.scaled_dot_product_attention(_to_copy_36,index_4,index_5,expand_8,False),) ;  _to_copy_36 = index_4 = index_5 = expand_8 = None
        getitem_36 = _scaled_dot_product_efficient_attention_default_1[0]
        view_69 = getitem_36.view(1, 4, 368, 32, 32) ;  getitem_36 = None
        permute_21 = view_69.permute(0, 2, 3, 1, 4) ;  view_69 = None
        clone_1 = torch.clone(permute_21,memory_format = torch.contiguous_format) ;  permute_21 = None
        _unsafe_view_1 = clone_1.view(1, 11776, 128) ;  clone_1 = None
        _to_copy_37 = arg29_1.to(dtype = torch.bfloat16) ;  arg29_1 = None
        _to_copy_38 = arg28_1.to(dtype = torch.bfloat16) ;  arg28_1 = None
        _to_copy_39 = view_19.to(dtype = torch.bfloat16) 
        view_70 = _to_copy_39.view(11776, 128) ;  _to_copy_39 = None
        t_12 = _to_copy_38.t() ;  _to_copy_38 = None
        addmm_2 = torch.addmm(_to_copy_37,view_70,t_12) ;  _to_copy_37 = view_70 = t_12 = None
        view_71 = addmm_2.view(1, 11776, 128) ;  addmm_2 = None
        sigmoid_2 = torch.sigmoid(view_71) ;  view_71 = None
        view_72 = sigmoid_2.view(11776, 128) ;  sigmoid_2 = None
        view_73 = view_72.view(1, 11776, 128) ;  view_72 = None
        mul_6 = torch.mul(_unsafe_view_1,view_73) ;  _unsafe_view_1 = view_73 = None
        _to_copy_40 = masked_fill_2.to(dtype = torch.float32) 
        native_layer_norm_default_5 = (torch.nn.functional.layer_norm(_to_copy_40,[128],None,None,0.1),) ;  _to_copy_40 = None
        getitem_40 = native_layer_norm_default_5[0]
        _to_copy_41 = arg10_1.to(dtype = torch.bfloat16) ;  arg10_1 = None
        _to_copy_42 = view_19.to(dtype = torch.bfloat16) 
        t_13 = _to_copy_41.t() ;  _to_copy_41 = None
        view_74 = _to_copy_42.view(11776, 128) ;  _to_copy_42 = None
        mm_10 = torch.mm(view_74,t_13) ;  view_74 = t_13 = None
        view_75 = mm_10.view(1, 11776, 256) ;  mm_10 = None
        split_tensor_4 = torch.split(view_75,128,dim = -1) ;  view_75 = None
        getitem_43 = split_tensor_4[0]
        getitem_44 = split_tensor_4[1];  split_tensor_4 = None
        add_15 = torch.add(getitem_43,1) ;  getitem_43 = None
        mul_7 = torch.mul(getitem_40,add_15) ;  getitem_40 = add_15 = None
        add_16 = torch.add(mul_7,getitem_44) ;  mul_7 = getitem_44 = None
        _to_copy_43 = arg11_1.to(dtype = torch.bfloat16) ;  arg11_1 = None
        _to_copy_44 = add_16.to(dtype = torch.bfloat16) ;  add_16 = None
        t_14 = _to_copy_43.t() ;  _to_copy_43 = None
        view_76 = _to_copy_44.view(11776, 128) ;  _to_copy_44 = None
        mm_11 = torch.mm(view_76,t_14) ;  view_76 = t_14 = None
        view_77 = mm_11.view(1, 11776, 512) ;  mm_11 = None
        split_tensor_5 = torch.split(view_77,256,dim = -1) ;  view_77 = None
        getitem_45 = split_tensor_5[0]
        getitem_46 = split_tensor_5[1];  split_tensor_5 = None
        silu_1 = torch.nn.functional.silu(getitem_45) ;  getitem_45 = None
        mul_8 = torch.mul(silu_1,getitem_46) ;  silu_1 = getitem_46 = None
        _to_copy_45 = arg14_1.to(dtype = torch.bfloat16) ;  arg14_1 = None
        _to_copy_46 = arg13_1.to(dtype = torch.bfloat16) ;  arg13_1 = None
        _to_copy_47 = view_19.to(dtype = torch.bfloat16) 
        view_78 = _to_copy_47.view(11776, 128) ;  _to_copy_47 = None
        t_15 = _to_copy_46.t() ;  _to_copy_46 = None
        addmm_3 = torch.addmm(_to_copy_45,view_78,t_15) ;  _to_copy_45 = view_78 = t_15 = None
        view_79 = addmm_3.view(1, 11776, 128) ;  addmm_3 = None
        sigmoid_3 = torch.sigmoid(view_79) ;  view_79 = None
        _to_copy_48 = arg12_1.to(dtype = torch.bfloat16) ;  arg12_1 = None
        t_16 = _to_copy_48.t() ;  _to_copy_48 = None
        view_80 = mul_8.view(11776, 256) ;  mul_8 = None
        mm_12 = torch.mm(view_80,t_16) ;  view_80 = t_16 = None
        view_81 = mm_12.view(1, 11776, 128) ;  mm_12 = None
        mul_9 = torch.mul(sigmoid_3,view_81) ;  sigmoid_3 = view_81 = None
        add_17 = torch.add(masked_fill_2,mul_9) ;  masked_fill_2 = mul_9 = None
        add_18 = torch.add(add_17,mul_6) ;  add_17 = mul_6 = None
        unsqueeze_30 = torch.unsqueeze(view_22,-1) ;  view_22 = None
        bitwise_not_4 = torch.bitwise_not(unsqueeze_30) ;  unsqueeze_30 = None
        masked_fill_4 = add_18.masked_fill(bitwise_not_4,0.0) ;  add_18 = bitwise_not_4 = None
        _to_copy_49 = getitem_3.to(dtype = torch.bfloat16) ;  getitem_3 = None
        _to_copy_50 = getitem_8.to(dtype = torch.bfloat16) ;  getitem_8 = None
        unsqueeze_31 = torch.unsqueeze(_to_copy_49,5) ;  _to_copy_49 = None
        permute_22 = unsqueeze_31.permute(0, 5, 1, 2, 3, 4) ;  unsqueeze_31 = None
        unsqueeze_32 = torch.unsqueeze(_to_copy_50,2) ;  _to_copy_50 = None
        unsqueeze_33 = torch.unsqueeze(unsqueeze_32,3) ;  unsqueeze_32 = None
        unsqueeze_34 = torch.unsqueeze(unsqueeze_33,4) ;  unsqueeze_33 = None
        unsqueeze_35 = torch.unsqueeze(unsqueeze_34,5) ;  unsqueeze_34 = None
        permute_23 = unsqueeze_35.permute(2, 0, 3, 4, 5, 1) ;  unsqueeze_35 = None
        permute_24 = permute_22.permute(2, 3, 4, 5, 0, 1) ;  permute_22 = None
        view_82 = permute_24.view(1, 1507328, 16) ;  permute_24 = None
        permute_25 = permute_23.permute(5, 0, 1, 2, 3, 4) ;  permute_23 = None
        view_83 = permute_25.view(1, 16, 4) ;  permute_25 = None
        bmm_4 = torch.bmm(view_82,view_83) ;  view_82 = view_83 = None
        view_84 = bmm_4.view(368, 32, 128, 1, 1, 4) ;  bmm_4 = None
        permute_26 = view_84.permute(4, 5, 0, 1, 2, 3) ;  view_84 = None
        view_85 = permute_26.view(1, 4, 368, 32, 128) ;  permute_26 = None
        view_86 = view_21.view(1, 1, 368, 32, 128) ;  view_21 = None
        bitwise_not_5 = torch.bitwise_not(view_86) ;  view_86 = None
        masked_fill_5 = view_85.masked_fill(bitwise_not_5,-10000) ;  view_85 = bitwise_not_5 = None
        _to_copy_51 = masked_fill_4.to(dtype = torch.float32) 
        native_layer_norm_default_6 = (torch.nn.functional.layer_norm(_to_copy_51,[128],None,None,0.1),) ;  _to_copy_51 = None
        getitem_47 = native_layer_norm_default_6[0]
        _to_copy_52 = arg31_1.to(dtype = torch.bfloat16) ;  arg31_1 = None
        _to_copy_53 = view_19.to(dtype = torch.bfloat16) 
        t_17 = _to_copy_52.t() ;  _to_copy_52 = None
        view_87 = _to_copy_53.view(11776, 128) ;  _to_copy_53 = None
        mm_13 = torch.mm(view_87,t_17) ;  view_87 = t_17 = None
        view_88 = mm_13.view(1, 11776, 256) ;  mm_13 = None
        split_tensor_6 = torch.split(view_88,128,dim = -1) ;  view_88 = None
        getitem_50 = split_tensor_6[0]
        getitem_51 = split_tensor_6[1];  split_tensor_6 = None
        add_19 = torch.add(getitem_50,1) ;  getitem_50 = None
        mul_10 = torch.mul(getitem_47,add_19) ;  getitem_47 = add_19 = None
        add_20 = torch.add(mul_10,getitem_51) ;  mul_10 = getitem_51 = None
        _to_copy_54 = add_20.to(dtype = torch.bfloat16) ;  add_20 = None
        _to_copy_55 = arg32_1.to(dtype = torch.bfloat16) ;  arg32_1 = None
        unsqueeze_36 = torch.unsqueeze(_to_copy_54,3) ;  _to_copy_54 = None
        unsqueeze_37 = torch.unsqueeze(unsqueeze_36,4) ;  unsqueeze_36 = None
        unsqueeze_38 = torch.unsqueeze(unsqueeze_37,5) ;  unsqueeze_37 = None
        permute_27 = unsqueeze_38.permute(3, 0, 4, 1, 5, 2) ;  unsqueeze_38 = None
        unsqueeze_39 = torch.unsqueeze(_to_copy_55,4) ;  _to_copy_55 = None
        unsqueeze_40 = torch.unsqueeze(unsqueeze_39,5) ;  unsqueeze_39 = None
        permute_28 = unsqueeze_40.permute(0, 4, 1, 5, 2, 3) ;  unsqueeze_40 = None
        permute_29 = permute_27.permute(3, 5, 0, 1, 2, 4) ;  permute_27 = None
        view_89 = permute_29.view(1, 11776, 128) ;  permute_29 = None
        permute_30 = permute_28.permute(5, 0, 1, 2, 4, 3) ;  permute_28 = None
        view_90 = permute_30.view(1, 128, 384) ;  permute_30 = None
        bmm_5 = torch.bmm(view_89,view_90) ;  view_89 = view_90 = None
        view_91 = bmm_5.view(11776, 1, 3, 1, 4, 32) ;  bmm_5 = None
        permute_31 = view_91.permute(2, 3, 4, 0, 5, 1) ;  view_91 = None
        view_92 = permute_31.view(3, 1, 4, 11776, 32) ;  permute_31 = None
        view_93 = view_92.view(3, 4, 11776, 32) ;  view_92 = None
        unbind_int_3 = torch.unbind(view_93) ;  view_93 = None
        getitem_52 = unbind_int_3[0]
        getitem_53 = unbind_int_3[1]
        getitem_54 = unbind_int_3[2];  unbind_int_3 = None
        unsqueeze_41 = torch.unsqueeze(arg30_1,0) ;  arg30_1 = None
        expand_9 = unsqueeze_41.expand(1, -1, -1) ;  unsqueeze_41 = None
        view_94 = expand_9.view(4, 1, 32) ;  expand_9 = None
        add_21 = torch.add(getitem_52,view_94) ;  getitem_52 = view_94 = None
        view_95 = add_21.view(4, 368, 32, 32) ;  add_21 = None
        slice_15 = getitem_53[0:] ;  getitem_53 = None
        slice_16 = slice_15[:, :, 0:] ;  slice_15 = None
        index_6 = slice_16[:,remainder] ;  slice_16 = None
        slice_17 = getitem_54[0:] ;  getitem_54 = None
        slice_18 = slice_17[:, :, 0:] ;  slice_17 = None
        index_7 = slice_18[:,remainder] ;  slice_18 = remainder = None
        view_96 = masked_fill_5.view(4, 368, 32, 128) ;  masked_fill_5 = None
        _to_copy_56 = view_95.to(dtype = torch.bfloat16) ;  view_95 = None
        expand_10 = view_96.expand(4, 368, 32, 128) ;  view_96 = None
        _scaled_dot_product_efficient_attention_default_2 = (torch.nn.functional.scaled_dot_product_attention(_to_copy_56,index_6,index_7,expand_10,False),) ;  _to_copy_56 = index_6 = index_7 = expand_10 = None
        getitem_55 = _scaled_dot_product_efficient_attention_default_2[0]
        view_97 = getitem_55.view(1, 4, 368, 32, 32) ;  getitem_55 = None
        permute_32 = view_97.permute(0, 2, 3, 1, 4) ;  view_97 = None
        clone_2 = torch.clone(permute_32,memory_format = torch.contiguous_format) ;  permute_32 = None
        _unsafe_view_2 = clone_2.view(1, 11776, 128) ;  clone_2 = None
        _to_copy_57 = arg34_1.to(dtype = torch.bfloat16) ;  arg34_1 = None
        _to_copy_58 = arg33_1.to(dtype = torch.bfloat16) ;  arg33_1 = None
        _to_copy_59 = view_19.to(dtype = torch.bfloat16) 
        view_98 = _to_copy_59.view(11776, 128) ;  _to_copy_59 = None
        t_18 = _to_copy_58.t() ;  _to_copy_58 = None
        addmm_4 = torch.addmm(_to_copy_57,view_98,t_18) ;  _to_copy_57 = view_98 = t_18 = None
        view_99 = addmm_4.view(1, 11776, 128) ;  addmm_4 = None
        sigmoid_4 = torch.sigmoid(view_99) ;  view_99 = None
        view_100 = sigmoid_4.view(11776, 128) ;  sigmoid_4 = None
        view_101 = view_100.view(1, 11776, 128) ;  view_100 = None
        mul_11 = torch.mul(_unsafe_view_2,view_101) ;  _unsafe_view_2 = view_101 = None
        _to_copy_60 = masked_fill_4.to(dtype = torch.float32) 
        native_layer_norm_default_7 = (torch.nn.functional.layer_norm(_to_copy_60,[128],None,None,0.1),) ;  _to_copy_60 = None
        getitem_59 = native_layer_norm_default_7[0]
        _to_copy_61 = arg15_1.to(dtype = torch.bfloat16) ;  arg15_1 = None
        _to_copy_62 = view_19.to(dtype = torch.bfloat16) 
        t_19 = _to_copy_61.t() ;  _to_copy_61 = None
        view_102 = _to_copy_62.view(11776, 128) ;  _to_copy_62 = None
        mm_14 = torch.mm(view_102,t_19) ;  view_102 = t_19 = None
        view_103 = mm_14.view(1, 11776, 256) ;  mm_14 = None
        split_tensor_7 = torch.split(view_103,128,dim = -1) ;  view_103 = None
        getitem_62 = split_tensor_7[0]
        getitem_63 = split_tensor_7[1];  split_tensor_7 = None
        add_22 = torch.add(getitem_62,1) ;  getitem_62 = None
        mul_12 = torch.mul(getitem_59,add_22) ;  getitem_59 = add_22 = None
        add_23 = torch.add(mul_12,getitem_63) ;  mul_12 = getitem_63 = None
        _to_copy_63 = arg16_1.to(dtype = torch.bfloat16) ;  arg16_1 = None
        _to_copy_64 = add_23.to(dtype = torch.bfloat16) ;  add_23 = None
        t_20 = _to_copy_63.t() ;  _to_copy_63 = None
        view_104 = _to_copy_64.view(11776, 128) ;  _to_copy_64 = None
        mm_15 = torch.mm(view_104,t_20) ;  view_104 = t_20 = None
        view_105 = mm_15.view(1, 11776, 512) ;  mm_15 = None
        split_tensor_8 = torch.split(view_105,256,dim = -1) ;  view_105 = None
        getitem_64 = split_tensor_8[0]
        getitem_65 = split_tensor_8[1];  split_tensor_8 = None
        silu_2 = torch.nn.functional.silu(getitem_64) ;  getitem_64 = None
        mul_13 = torch.mul(silu_2,getitem_65) ;  silu_2 = getitem_65 = None
        _to_copy_65 = arg19_1.to(dtype = torch.bfloat16) ;  arg19_1 = None
        _to_copy_66 = arg18_1.to(dtype = torch.bfloat16) ;  arg18_1 = None
        _to_copy_67 = view_19.to(dtype = torch.bfloat16) ;  view_19 = None
        view_106 = _to_copy_67.view(11776, 128) ;  _to_copy_67 = None
        t_21 = _to_copy_66.t() ;  _to_copy_66 = None
        addmm_5 = torch.addmm(_to_copy_65,view_106,t_21) ;  _to_copy_65 = view_106 = t_21 = None
        view_107 = addmm_5.view(1, 11776, 128) ;  addmm_5 = None
        sigmoid_5 = torch.sigmoid(view_107) ;  view_107 = None
        _to_copy_68 = arg17_1.to(dtype = torch.bfloat16) ;  arg17_1 = None
        t_22 = _to_copy_68.t() ;  _to_copy_68 = None
        view_108 = mul_13.view(11776, 256) ;  mul_13 = None
        mm_16 = torch.mm(view_108,t_22) ;  view_108 = t_22 = None
        view_109 = mm_16.view(1, 11776, 128) ;  mm_16 = None
        mul_14 = torch.mul(sigmoid_5,view_109) ;  sigmoid_5 = view_109 = None
        add_24 = torch.add(masked_fill_4,mul_14) ;  masked_fill_4 = mul_14 = None
        add_25 = torch.add(add_24,mul_11) ;  add_24 = mul_11 = None
        view_110 = add_25.view(1, 1, 11776, 128) ;  add_25 = None
        _to_copy_69 = arg38_1.to(dtype = torch.bfloat16) ;  arg38_1 = None
        t_23 = _to_copy_69.t() ;  _to_copy_69 = None
        view_111 = view_110.view(11776, 128) ;  view_110 = None
        mm_17 = torch.mm(view_111,t_23) ;  view_111 = t_23 = None
        view_112 = mm_17.view(1, 1, 11776, 384) ;  mm_17 = None
        relu_3 = torch.relu(view_112) ;  view_112 = None
        view_113 = relu_3.view(11776, 384) ;  relu_3 = None
        unsqueeze_42 = torch.unsqueeze(arg50_1,1) ;  arg50_1 = None
        expand_11 = unsqueeze_42.expand(-1, 1, -1) ;  unsqueeze_42 = None
        view_116 = expand_11.view(1, 11776) ;  expand_11 = None
        unsqueeze_43 = torch.unsqueeze(arg51_1,1) ;  arg51_1 = None
        expand_12 = unsqueeze_43.expand(-1, 1, -1) ;  unsqueeze_43 = None
        view_117 = expand_12.view(1, 11776) ;  expand_12 = None
        view_118 = view_113.view(1, 1, 11776, 384) ;  view_113 = None
        view_119 = view_118.view(1, 11776, 384) ;  view_118 = None
        new_zeros = view_119.new_zeros((1,512,384), pin_memory = False)
        new_zeros_1 = view_119.new_zeros((1,512), pin_memory = False)
        unsqueeze_44 = torch.unsqueeze(view_117,2) 
        expand_13 = unsqueeze_44.expand(-1, -1, 384) ;  unsqueeze_44 = None
        unsqueeze_45 = torch.unsqueeze(view_116,-1) 
        mul_15 = torch.mul(view_119,unsqueeze_45) ;  view_119 = unsqueeze_45 = None
        scatter_reduce = torch.scatter_reduce(new_zeros,1,expand_13,mul_15,'sum') ;  new_zeros = expand_13 = mul_15 = None
        _to_copy_70 = view_116.to(dtype = torch.bfloat16) ;  view_116 = None
        scatter_reduce_1 = torch.scatter_reduce(new_zeros_1,1,view_117,_to_copy_70,'sum') ;  new_zeros_1 = view_117 = _to_copy_70 = None
        unsqueeze_46 = torch.unsqueeze(scatter_reduce_1,-1) ;  scatter_reduce_1 = None
        clamp = torch.clamp(unsqueeze_46,min = 1) ;  unsqueeze_46 = None
        div = torch.div(scatter_reduce,clamp) ;  scatter_reduce = clamp = None
        view_120 = div.view(1, 1, 512, 384) ;  div = None
        view_121 = view_120.view(1, 512, 384) ;  view_120 = None
        cat = torch.cat([view_121,arg43_1],dim = -1) ;  view_121 = arg43_1 = None
        _to_copy_71 = arg40_1.to(dtype = torch.bfloat16) ;  arg40_1 = None
        t_24 = _to_copy_71.t() ;  _to_copy_71 = None
        view_122 = cat.view(512, 768) 
        mm_18 = torch.mm(view_122,t_24) ;  view_122 = t_24 = None
        view_123 = mm_18.view(1, 512, 384) ;  mm_18 = None
        _to_copy_72 = arg39_1.to(dtype = torch.bfloat16) ;  arg39_1 = None
        t_25 = _to_copy_72.t() ;  _to_copy_72 = None
        view_124 = cat.view(512, 768) ;  cat = None
        mm_19 = torch.mm(view_124,t_25) ;  view_124 = t_25 = None
        view_125 = mm_19.view(1, 512, 384) ;  mm_19 = None
        _to_copy_73 = arg42_1.to(dtype = torch.bfloat16) ;  arg42_1 = None
        t_26 = _to_copy_73.t() ;  _to_copy_73 = None
        view_126 = view_125.view(512, 384) 
        mm_20 = torch.mm(view_126,t_26) ;  view_126 = t_26 = None
        view_127 = mm_20.view(1, 512, 512) ;  mm_20 = None
        split_tensor_9 = torch.split(view_127,256,dim = -1) ;  view_127 = None
        getitem_66 = split_tensor_9[0]
        getitem_67 = split_tensor_9[1];  split_tensor_9 = None
        view_128 = getitem_66.view(1, 512, 1, 256) ;  getitem_66 = None
        add_26 = torch.add(arg44_1,view_128) ;  arg44_1 = view_128 = None
        view_129 = getitem_67.view(1, 1, 512, 256) ;  getitem_67 = None
        add_27 = torch.add(add_26,view_129) ;  add_26 = view_129 = None
        _to_copy_74 = arg41_1.to(dtype = torch.bfloat16) ;  arg41_1 = None
        t_27 = _to_copy_74.t() ;  _to_copy_74 = None
        view_130 = add_27.view(262144, 256) ;  add_27 = None
        mm_21 = torch.mm(view_130,t_27) ;  view_130 = t_27 = None
        view_131 = mm_21.view(1, 512, 512, 256) ;  mm_21 = None
        return (view_125, view_123, view_131)
        
    # To see more debug info, please use `graph_module.print_readable()`