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

        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.bfloat16);  arg0_1 = None
        t = torch.ops.aten.t.default(_to_copy);  _to_copy = None
        view = torch.ops.aten.view.default(arg45_1, [11776, 128]);  arg45_1 = None
        mm = torch.ops.aten.mm.default(view, t);  view = t = None
        view_1 = torch.ops.aten.view.default(mm, [1, 11776, 128]);  mm = None
        unsqueeze = torch.ops.aten.unsqueeze.default(view_1, 1)
        expand = torch.ops.aten.expand.default(unsqueeze, [-1, 1, -1, -1]);  unsqueeze = None
        _to_copy_1 = torch.ops.aten._to_copy.default(view_1, dtype = torch.float32);  view_1 = None
        native_layer_norm_default = torch.ops.aten.native_layer_norm.default(_to_copy_1, [128], None, None, 1e-05);  _to_copy_1 = None
        getitem = native_layer_norm_default[0]
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(getitem, 1)
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [-1, 1, -1, -1]);  unsqueeze_1 = None
        slice_1 = torch.ops.aten.slice.Tensor(expand_1, dim = 0, start = 0, end = 9223372036854775807)
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, dim = 1, start = 0, end = 9223372036854775807);  slice_1 = None
        index = torch.ops.aten.index.Tensor(slice_2, [None, None, arg48_1]);  slice_2 = arg48_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(expand_1, dim = 0, start = 0, end = 9223372036854775807);  expand_1 = None
        slice_4 = torch.ops.aten.slice.Tensor(slice_3, dim = 1, start = 0, end = 9223372036854775807);  slice_3 = None
        index_1 = torch.ops.aten.index.Tensor(slice_4, [None, None, arg49_1]);  slice_4 = arg49_1 = None
        relu = torch.ops.aten.relu.default(index);  index = None
        _to_copy_2 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.bfloat16);  arg1_1 = None
        _to_copy_3 = torch.ops.aten._to_copy.default(relu, dtype = torch.bfloat16);  relu = None
        t_1 = torch.ops.aten.t.default(_to_copy_2);  _to_copy_2 = None
        view_2 = torch.ops.aten.view.default(_to_copy_3, [11776, 128]);  _to_copy_3 = None
        mm_1 = torch.ops.aten.mm.default(view_2, t_1);  view_2 = t_1 = None
        view_3 = torch.ops.aten.view.default(mm_1, [1, 1, 368, 32, 16]);  mm_1 = None
        relu_1 = torch.ops.aten.relu.default(index_1);  index_1 = None
        _to_copy_4 = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.bfloat16);  arg2_1 = None
        _to_copy_5 = torch.ops.aten._to_copy.default(relu_1, dtype = torch.bfloat16);  relu_1 = None
        t_2 = torch.ops.aten.t.default(_to_copy_4);  _to_copy_4 = None
        view_4 = torch.ops.aten.view.default(_to_copy_5, [47104, 128]);  _to_copy_5 = None
        mm_2 = torch.ops.aten.mm.default(view_4, t_2);  view_4 = t_2 = None
        view_5 = torch.ops.aten.view.default(mm_2, [1, 1, 368, 128, 16]);  mm_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg46_1, 1);  arg46_1 = None
        view_6 = torch.ops.aten.view.default(view_3, [1, 1, 368, 32, 1, 16]);  view_3 = None
        add = torch.ops.aten.add.Tensor(unsqueeze_2, view_6);  unsqueeze_2 = view_6 = None
        view_7 = torch.ops.aten.view.default(view_5, [1, 1, 368, 1, 128, 16]);  view_5 = None
        add_1 = torch.ops.aten.add.Tensor(add, view_7);  add = view_7 = None
        _to_copy_6 = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.bfloat16);  arg3_1 = None
        t_3 = torch.ops.aten.t.default(_to_copy_6);  _to_copy_6 = None
        view_8 = torch.ops.aten.view.default(add_1, [1507328, 16])
        mm_3 = torch.ops.aten.mm.default(view_8, t_3);  view_8 = t_3 = None
        view_9 = torch.ops.aten.view.default(mm_3, [1, 1, 368, 32, 128, 16]);  mm_3 = None
        relu_2 = torch.ops.aten.relu.default(view_9);  view_9 = None
        view_10 = torch.ops.aten.view.default(relu_2, [1507328, 16]);  relu_2 = None
        _to_copy_7 = torch.ops.aten._to_copy.default(arg4_1, dtype = torch.bfloat16);  arg4_1 = None
        t_4 = torch.ops.aten.t.default(_to_copy_7);  _to_copy_7 = None
        view_13 = torch.ops.aten.view.default(view_10, [1, 1, 368, 32, 128, 16]);  view_10 = None
        view_14 = torch.ops.aten.view.default(view_13, [1507328, 16]);  view_13 = None
        mm_4 = torch.ops.aten.mm.default(view_14, t_4);  view_14 = t_4 = None
        view_15 = torch.ops.aten.view.default(mm_4, [1, 1, 368, 32, 128, 16]);  mm_4 = None
        add_2 = torch.ops.aten.add.Tensor(view_15, add_1);  view_15 = add_1 = None
        view_16 = torch.ops.aten.view.default(add_2, [1507328, 16]);  add_2 = None
        view_18 = torch.ops.aten.view.default(expand, [1, 11776, 128]);  expand = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem, 1);  getitem = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_3, [-1, 1, -1, -1]);  unsqueeze_3 = None
        view_19 = torch.ops.aten.view.default(expand_2, [1, 11776, 128]);  expand_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg47_1, 1);  arg47_1 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_4, [-1, 1, -1, -1, -1]);  unsqueeze_4 = None
        view_21 = torch.ops.aten.view.default(expand_3, [1, 368, 32, 128]);  expand_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(arg50_1, 1)
        expand_4 = torch.ops.aten.expand.default(unsqueeze_5, [-1, 1, -1]);  unsqueeze_5 = None
        view_22 = torch.ops.aten.view.default(expand_4, [1, 11776]);  expand_4 = None
        arange = torch.ops.aten.arange.default(11776, device = self.device, pin_memory = False)
        view_23 = torch.ops.aten.view.default(arange, [368, 32]);  arange = None
        slice_5 = torch.ops.aten.slice.Tensor(view_23, dim = 0, start = 0, end = 9223372036854775807);  view_23 = None
        slice_6 = torch.ops.aten.slice.Tensor(slice_5, dim = 1, start = 0, end = 1);  slice_5 = None
        add_3 = torch.ops.aten.add.Tensor(slice_6, -48);  slice_6 = None
        arange_1 = torch.ops.aten.arange.default(128, device = self.device, pin_memory = False)
        add_4 = torch.ops.aten.add.Tensor(add_3, arange_1);  add_3 = arange_1 = None
        remainder = torch.ops.aten.remainder.Scalar(add_4, 11776);  add_4 = None
        view_24 = torch.ops.aten.view.default(view_16, [1, 1, 368, 32, 128, 16]);  view_16 = None
        view_25 = torch.ops.aten.view.default(view_24, [1, 368, 32, 128, 16]);  view_24 = None
        _to_copy_8 = torch.ops.aten._to_copy.default(view_25, dtype = torch.float32);  view_25 = None
        native_layer_norm_default_1 = torch.ops.aten.native_layer_norm.default(_to_copy_8, [16], arg35_1, arg36_1, 1e-05);  _to_copy_8 = arg35_1 = arg36_1 = None
        getitem_3 = native_layer_norm_default_1[0]
        unbind_int = torch.ops.aten.unbind.int(arg37_1);  arg37_1 = None
        getitem_6 = unbind_int[0]
        getitem_7 = unbind_int[1]
        getitem_8 = unbind_int[2];  unbind_int = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(view_22, -1)
        bitwise_not = torch.ops.aten.bitwise_not.default(unsqueeze_6);  unsqueeze_6 = None
        masked_fill = torch.ops.aten.masked_fill.Scalar(view_18, bitwise_not, 0.0);  view_18 = bitwise_not = None
        _to_copy_9 = torch.ops.aten._to_copy.default(getitem_3, dtype = torch.bfloat16)
        _to_copy_10 = torch.ops.aten._to_copy.default(getitem_6, dtype = torch.bfloat16);  getitem_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(_to_copy_9, 5);  _to_copy_9 = None
        permute = torch.ops.aten.permute.default(unsqueeze_7, [0, 5, 1, 2, 3, 4]);  unsqueeze_7 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(_to_copy_10, 2);  _to_copy_10 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 5);  unsqueeze_10 = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze_11, [2, 0, 3, 4, 5, 1]);  unsqueeze_11 = None
        permute_2 = torch.ops.aten.permute.default(permute, [2, 3, 4, 5, 0, 1]);  permute = None
        view_26 = torch.ops.aten.view.default(permute_2, [1, 1507328, 16]);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [5, 0, 1, 2, 3, 4]);  permute_1 = None
        view_27 = torch.ops.aten.view.default(permute_3, [1, 16, 4]);  permute_3 = None
        bmm = torch.ops.aten.bmm.default(view_26, view_27);  view_26 = view_27 = None
        view_28 = torch.ops.aten.view.default(bmm, [368, 32, 128, 1, 1, 4]);  bmm = None
        permute_4 = torch.ops.aten.permute.default(view_28, [4, 5, 0, 1, 2, 3]);  view_28 = None
        view_29 = torch.ops.aten.view.default(permute_4, [1, 4, 368, 32, 128]);  permute_4 = None
        view_30 = torch.ops.aten.view.default(view_21, [1, 1, 368, 32, 128])
        bitwise_not_1 = torch.ops.aten.bitwise_not.default(view_30);  view_30 = None
        masked_fill_1 = torch.ops.aten.masked_fill.Scalar(view_29, bitwise_not_1, -10000);  view_29 = bitwise_not_1 = None
        _to_copy_11 = torch.ops.aten._to_copy.default(masked_fill, dtype = torch.float32)
        native_layer_norm_default_2 = torch.ops.aten.native_layer_norm.default(_to_copy_11, [128], None, None, 0.1);  _to_copy_11 = None
        getitem_9 = native_layer_norm_default_2[0]
        _to_copy_12 = torch.ops.aten._to_copy.default(arg21_1, dtype = torch.bfloat16);  arg21_1 = None
        _to_copy_13 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        t_5 = torch.ops.aten.t.default(_to_copy_12);  _to_copy_12 = None
        view_31 = torch.ops.aten.view.default(_to_copy_13, [11776, 128]);  _to_copy_13 = None
        mm_5 = torch.ops.aten.mm.default(view_31, t_5);  view_31 = t_5 = None
        view_32 = torch.ops.aten.view.default(mm_5, [1, 11776, 256]);  mm_5 = None
        split_tensor = torch.ops.aten.split.Tensor(view_32, 128, dim = -1);  view_32 = None
        getitem_12 = split_tensor[0]
        getitem_13 = split_tensor[1];  split_tensor = None
        add_5 = torch.ops.aten.add.Tensor(getitem_12, 1);  getitem_12 = None
        mul = torch.ops.aten.mul.Tensor(getitem_9, add_5);  getitem_9 = add_5 = None
        add_6 = torch.ops.aten.add.Tensor(mul, getitem_13);  mul = getitem_13 = None
        _to_copy_14 = torch.ops.aten._to_copy.default(add_6, dtype = torch.bfloat16);  add_6 = None
        _to_copy_15 = torch.ops.aten._to_copy.default(arg22_1, dtype = torch.bfloat16);  arg22_1 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(_to_copy_14, 3);  _to_copy_14 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 4);  unsqueeze_12 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 5);  unsqueeze_13 = None
        permute_5 = torch.ops.aten.permute.default(unsqueeze_14, [3, 0, 4, 1, 5, 2]);  unsqueeze_14 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(_to_copy_15, 4);  _to_copy_15 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(unsqueeze_15, 5);  unsqueeze_15 = None
        permute_6 = torch.ops.aten.permute.default(unsqueeze_16, [0, 4, 1, 5, 2, 3]);  unsqueeze_16 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [3, 5, 0, 1, 2, 4]);  permute_5 = None
        view_33 = torch.ops.aten.view.default(permute_7, [1, 11776, 128]);  permute_7 = None
        permute_8 = torch.ops.aten.permute.default(permute_6, [5, 0, 1, 2, 4, 3]);  permute_6 = None
        view_34 = torch.ops.aten.view.default(permute_8, [1, 128, 384]);  permute_8 = None
        bmm_1 = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
        view_35 = torch.ops.aten.view.default(bmm_1, [11776, 1, 3, 1, 4, 32]);  bmm_1 = None
        permute_9 = torch.ops.aten.permute.default(view_35, [2, 3, 4, 0, 5, 1]);  view_35 = None
        view_36 = torch.ops.aten.view.default(permute_9, [3, 1, 4, 11776, 32]);  permute_9 = None
        view_37 = torch.ops.aten.view.default(view_36, [3, 4, 11776, 32]);  view_36 = None
        unbind_int_1 = torch.ops.aten.unbind.int(view_37);  view_37 = None
        getitem_14 = unbind_int_1[0]
        getitem_15 = unbind_int_1[1]
        getitem_16 = unbind_int_1[2];  unbind_int_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(arg20_1, 0);  arg20_1 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_17, [1, -1, -1]);  unsqueeze_17 = None
        view_38 = torch.ops.aten.view.default(expand_5, [4, 1, 32]);  expand_5 = None
        add_7 = torch.ops.aten.add.Tensor(getitem_14, view_38);  getitem_14 = view_38 = None
        view_39 = torch.ops.aten.view.default(add_7, [4, 368, 32, 32]);  add_7 = None
        slice_7 = torch.ops.aten.slice.Tensor(getitem_15, dim = 0, start = 0, end = 9223372036854775807);  getitem_15 = None
        slice_8 = torch.ops.aten.slice.Tensor(slice_7, dim = 2, start = 0, end = 9223372036854775807);  slice_7 = None
        index_2 = torch.ops.aten.index.Tensor(slice_8, [None, remainder]);  slice_8 = None
        slice_9 = torch.ops.aten.slice.Tensor(getitem_16, dim = 0, start = 0, end = 9223372036854775807);  getitem_16 = None
        slice_10 = torch.ops.aten.slice.Tensor(slice_9, dim = 2, start = 0, end = 9223372036854775807);  slice_9 = None
        index_3 = torch.ops.aten.index.Tensor(slice_10, [None, remainder]);  slice_10 = None
        view_40 = torch.ops.aten.view.default(masked_fill_1, [4, 368, 32, 128]);  masked_fill_1 = None
        _to_copy_16 = torch.ops.aten._to_copy.default(view_39, dtype = torch.bfloat16);  view_39 = None
        expand_6 = torch.ops.aten.expand.default(view_40, [4, 368, 32, 128]);  view_40 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(_to_copy_16, index_2, index_3, expand_6, False);  _to_copy_16 = index_2 = index_3 = expand_6 = None
        getitem_17 = _scaled_dot_product_efficient_attention_default[0]
        view_41 = torch.ops.aten.view.default(getitem_17, [1, 4, 368, 32, 32]);  getitem_17 = None
        permute_10 = torch.ops.aten.permute.default(view_41, [0, 2, 3, 1, 4]);  view_41 = None
        clone = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [1, 11776, 128]);  clone = None
        _to_copy_17 = torch.ops.aten._to_copy.default(arg24_1, dtype = torch.bfloat16);  arg24_1 = None
        _to_copy_18 = torch.ops.aten._to_copy.default(arg23_1, dtype = torch.bfloat16);  arg23_1 = None
        _to_copy_19 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        view_42 = torch.ops.aten.view.default(_to_copy_19, [11776, 128]);  _to_copy_19 = None
        t_6 = torch.ops.aten.t.default(_to_copy_18);  _to_copy_18 = None
        addmm = torch.ops.aten.addmm.default(_to_copy_17, view_42, t_6);  _to_copy_17 = view_42 = t_6 = None
        view_43 = torch.ops.aten.view.default(addmm, [1, 11776, 128]);  addmm = None
        sigmoid = torch.ops.aten.sigmoid.default(view_43);  view_43 = None
        view_44 = torch.ops.aten.view.default(sigmoid, [11776, 128]);  sigmoid = None
        view_45 = torch.ops.aten.view.default(view_44, [1, 11776, 128]);  view_44 = None
        mul_1 = torch.ops.aten.mul.Tensor(_unsafe_view, view_45);  _unsafe_view = view_45 = None
        _to_copy_20 = torch.ops.aten._to_copy.default(masked_fill, dtype = torch.float32)
        native_layer_norm_default_3 = torch.ops.aten.native_layer_norm.default(_to_copy_20, [128], None, None, 0.1);  _to_copy_20 = None
        getitem_21 = native_layer_norm_default_3[0]
        _to_copy_21 = torch.ops.aten._to_copy.default(arg5_1, dtype = torch.bfloat16);  arg5_1 = None
        _to_copy_22 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        t_7 = torch.ops.aten.t.default(_to_copy_21);  _to_copy_21 = None
        view_46 = torch.ops.aten.view.default(_to_copy_22, [11776, 128]);  _to_copy_22 = None
        mm_6 = torch.ops.aten.mm.default(view_46, t_7);  view_46 = t_7 = None
        view_47 = torch.ops.aten.view.default(mm_6, [1, 11776, 256]);  mm_6 = None
        split_tensor_1 = torch.ops.aten.split.Tensor(view_47, 128, dim = -1);  view_47 = None
        getitem_24 = split_tensor_1[0]
        getitem_25 = split_tensor_1[1];  split_tensor_1 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_24, 1);  getitem_24 = None
        mul_2 = torch.ops.aten.mul.Tensor(getitem_21, add_8);  getitem_21 = add_8 = None
        add_9 = torch.ops.aten.add.Tensor(mul_2, getitem_25);  mul_2 = getitem_25 = None
        _to_copy_23 = torch.ops.aten._to_copy.default(arg6_1, dtype = torch.bfloat16);  arg6_1 = None
        _to_copy_24 = torch.ops.aten._to_copy.default(add_9, dtype = torch.bfloat16);  add_9 = None
        t_8 = torch.ops.aten.t.default(_to_copy_23);  _to_copy_23 = None
        view_48 = torch.ops.aten.view.default(_to_copy_24, [11776, 128]);  _to_copy_24 = None
        mm_7 = torch.ops.aten.mm.default(view_48, t_8);  view_48 = t_8 = None
        view_49 = torch.ops.aten.view.default(mm_7, [1, 11776, 512]);  mm_7 = None
        split_tensor_2 = torch.ops.aten.split.Tensor(view_49, 256, dim = -1);  view_49 = None
        getitem_26 = split_tensor_2[0]
        getitem_27 = split_tensor_2[1];  split_tensor_2 = None
        silu = torch.ops.aten.silu.default(getitem_26);  getitem_26 = None
        mul_3 = torch.ops.aten.mul.Tensor(silu, getitem_27);  silu = getitem_27 = None
        _to_copy_25 = torch.ops.aten._to_copy.default(arg9_1, dtype = torch.bfloat16);  arg9_1 = None
        _to_copy_26 = torch.ops.aten._to_copy.default(arg8_1, dtype = torch.bfloat16);  arg8_1 = None
        _to_copy_27 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        view_50 = torch.ops.aten.view.default(_to_copy_27, [11776, 128]);  _to_copy_27 = None
        t_9 = torch.ops.aten.t.default(_to_copy_26);  _to_copy_26 = None
        addmm_1 = torch.ops.aten.addmm.default(_to_copy_25, view_50, t_9);  _to_copy_25 = view_50 = t_9 = None
        view_51 = torch.ops.aten.view.default(addmm_1, [1, 11776, 128]);  addmm_1 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(view_51);  view_51 = None
        _to_copy_28 = torch.ops.aten._to_copy.default(arg7_1, dtype = torch.bfloat16);  arg7_1 = None
        t_10 = torch.ops.aten.t.default(_to_copy_28);  _to_copy_28 = None
        view_52 = torch.ops.aten.view.default(mul_3, [11776, 256]);  mul_3 = None
        mm_8 = torch.ops.aten.mm.default(view_52, t_10);  view_52 = t_10 = None
        view_53 = torch.ops.aten.view.default(mm_8, [1, 11776, 128]);  mm_8 = None
        mul_4 = torch.ops.aten.mul.Tensor(sigmoid_1, view_53);  sigmoid_1 = view_53 = None
        add_10 = torch.ops.aten.add.Tensor(masked_fill, mul_4);  masked_fill = mul_4 = None
        add_11 = torch.ops.aten.add.Tensor(add_10, mul_1);  add_10 = mul_1 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(view_22, -1)
        bitwise_not_2 = torch.ops.aten.bitwise_not.default(unsqueeze_18);  unsqueeze_18 = None
        masked_fill_2 = torch.ops.aten.masked_fill.Scalar(add_11, bitwise_not_2, 0.0);  add_11 = bitwise_not_2 = None
        _to_copy_29 = torch.ops.aten._to_copy.default(getitem_3, dtype = torch.bfloat16)
        _to_copy_30 = torch.ops.aten._to_copy.default(getitem_7, dtype = torch.bfloat16);  getitem_7 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(_to_copy_29, 5);  _to_copy_29 = None
        permute_11 = torch.ops.aten.permute.default(unsqueeze_19, [0, 5, 1, 2, 3, 4]);  unsqueeze_19 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(_to_copy_30, 2);  _to_copy_30 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, 3);  unsqueeze_20 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(unsqueeze_21, 4);  unsqueeze_21 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, 5);  unsqueeze_22 = None
        permute_12 = torch.ops.aten.permute.default(unsqueeze_23, [2, 0, 3, 4, 5, 1]);  unsqueeze_23 = None
        permute_13 = torch.ops.aten.permute.default(permute_11, [2, 3, 4, 5, 0, 1]);  permute_11 = None
        view_54 = torch.ops.aten.view.default(permute_13, [1, 1507328, 16]);  permute_13 = None
        permute_14 = torch.ops.aten.permute.default(permute_12, [5, 0, 1, 2, 3, 4]);  permute_12 = None
        view_55 = torch.ops.aten.view.default(permute_14, [1, 16, 4]);  permute_14 = None
        bmm_2 = torch.ops.aten.bmm.default(view_54, view_55);  view_54 = view_55 = None
        view_56 = torch.ops.aten.view.default(bmm_2, [368, 32, 128, 1, 1, 4]);  bmm_2 = None
        permute_15 = torch.ops.aten.permute.default(view_56, [4, 5, 0, 1, 2, 3]);  view_56 = None
        view_57 = torch.ops.aten.view.default(permute_15, [1, 4, 368, 32, 128]);  permute_15 = None
        view_58 = torch.ops.aten.view.default(view_21, [1, 1, 368, 32, 128])
        bitwise_not_3 = torch.ops.aten.bitwise_not.default(view_58);  view_58 = None
        masked_fill_3 = torch.ops.aten.masked_fill.Scalar(view_57, bitwise_not_3, -10000);  view_57 = bitwise_not_3 = None
        _to_copy_31 = torch.ops.aten._to_copy.default(masked_fill_2, dtype = torch.float32)
        native_layer_norm_default_4 = torch.ops.aten.native_layer_norm.default(_to_copy_31, [128], None, None, 0.1);  _to_copy_31 = None
        getitem_28 = native_layer_norm_default_4[0]
        _to_copy_32 = torch.ops.aten._to_copy.default(arg26_1, dtype = torch.bfloat16);  arg26_1 = None
        _to_copy_33 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        t_11 = torch.ops.aten.t.default(_to_copy_32);  _to_copy_32 = None
        view_59 = torch.ops.aten.view.default(_to_copy_33, [11776, 128]);  _to_copy_33 = None
        mm_9 = torch.ops.aten.mm.default(view_59, t_11);  view_59 = t_11 = None
        view_60 = torch.ops.aten.view.default(mm_9, [1, 11776, 256]);  mm_9 = None
        split_tensor_3 = torch.ops.aten.split.Tensor(view_60, 128, dim = -1);  view_60 = None
        getitem_31 = split_tensor_3[0]
        getitem_32 = split_tensor_3[1];  split_tensor_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_31, 1);  getitem_31 = None
        mul_5 = torch.ops.aten.mul.Tensor(getitem_28, add_12);  getitem_28 = add_12 = None
        add_13 = torch.ops.aten.add.Tensor(mul_5, getitem_32);  mul_5 = getitem_32 = None
        _to_copy_34 = torch.ops.aten._to_copy.default(add_13, dtype = torch.bfloat16);  add_13 = None
        _to_copy_35 = torch.ops.aten._to_copy.default(arg27_1, dtype = torch.bfloat16);  arg27_1 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(_to_copy_34, 3);  _to_copy_34 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, 4);  unsqueeze_24 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(unsqueeze_25, 5);  unsqueeze_25 = None
        permute_16 = torch.ops.aten.permute.default(unsqueeze_26, [3, 0, 4, 1, 5, 2]);  unsqueeze_26 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(_to_copy_35, 4);  _to_copy_35 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(unsqueeze_27, 5);  unsqueeze_27 = None
        permute_17 = torch.ops.aten.permute.default(unsqueeze_28, [0, 4, 1, 5, 2, 3]);  unsqueeze_28 = None
        permute_18 = torch.ops.aten.permute.default(permute_16, [3, 5, 0, 1, 2, 4]);  permute_16 = None
        view_61 = torch.ops.aten.view.default(permute_18, [1, 11776, 128]);  permute_18 = None
        permute_19 = torch.ops.aten.permute.default(permute_17, [5, 0, 1, 2, 4, 3]);  permute_17 = None
        view_62 = torch.ops.aten.view.default(permute_19, [1, 128, 384]);  permute_19 = None
        bmm_3 = torch.ops.aten.bmm.default(view_61, view_62);  view_61 = view_62 = None
        view_63 = torch.ops.aten.view.default(bmm_3, [11776, 1, 3, 1, 4, 32]);  bmm_3 = None
        permute_20 = torch.ops.aten.permute.default(view_63, [2, 3, 4, 0, 5, 1]);  view_63 = None
        view_64 = torch.ops.aten.view.default(permute_20, [3, 1, 4, 11776, 32]);  permute_20 = None
        view_65 = torch.ops.aten.view.default(view_64, [3, 4, 11776, 32]);  view_64 = None
        unbind_int_2 = torch.ops.aten.unbind.int(view_65);  view_65 = None
        getitem_33 = unbind_int_2[0]
        getitem_34 = unbind_int_2[1]
        getitem_35 = unbind_int_2[2];  unbind_int_2 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(arg25_1, 0);  arg25_1 = None
        expand_7 = torch.ops.aten.expand.default(unsqueeze_29, [1, -1, -1]);  unsqueeze_29 = None
        view_66 = torch.ops.aten.view.default(expand_7, [4, 1, 32]);  expand_7 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_33, view_66);  getitem_33 = view_66 = None
        view_67 = torch.ops.aten.view.default(add_14, [4, 368, 32, 32]);  add_14 = None
        slice_11 = torch.ops.aten.slice.Tensor(getitem_34, dim = 0, start = 0, end = 9223372036854775807);  getitem_34 = None
        slice_12 = torch.ops.aten.slice.Tensor(slice_11, dim = 2, start = 0, end = 9223372036854775807);  slice_11 = None
        index_4 = torch.ops.aten.index.Tensor(slice_12, [None, remainder]);  slice_12 = None
        slice_13 = torch.ops.aten.slice.Tensor(getitem_35, dim = 0, start = 0, end = 9223372036854775807);  getitem_35 = None
        slice_14 = torch.ops.aten.slice.Tensor(slice_13, dim = 2, start = 0, end = 9223372036854775807);  slice_13 = None
        index_5 = torch.ops.aten.index.Tensor(slice_14, [None, remainder]);  slice_14 = None
        view_68 = torch.ops.aten.view.default(masked_fill_3, [4, 368, 32, 128]);  masked_fill_3 = None
        _to_copy_36 = torch.ops.aten._to_copy.default(view_67, dtype = torch.bfloat16);  view_67 = None
        expand_8 = torch.ops.aten.expand.default(view_68, [4, 368, 32, 128]);  view_68 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(_to_copy_36, index_4, index_5, expand_8, False);  _to_copy_36 = index_4 = index_5 = expand_8 = None
        getitem_36 = _scaled_dot_product_efficient_attention_default_1[0]
        view_69 = torch.ops.aten.view.default(getitem_36, [1, 4, 368, 32, 32]);  getitem_36 = None
        permute_21 = torch.ops.aten.permute.default(view_69, [0, 2, 3, 1, 4]);  view_69 = None
        clone_1 = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_1, [1, 11776, 128]);  clone_1 = None
        _to_copy_37 = torch.ops.aten._to_copy.default(arg29_1, dtype = torch.bfloat16);  arg29_1 = None
        _to_copy_38 = torch.ops.aten._to_copy.default(arg28_1, dtype = torch.bfloat16);  arg28_1 = None
        _to_copy_39 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        view_70 = torch.ops.aten.view.default(_to_copy_39, [11776, 128]);  _to_copy_39 = None
        t_12 = torch.ops.aten.t.default(_to_copy_38);  _to_copy_38 = None
        addmm_2 = torch.ops.aten.addmm.default(_to_copy_37, view_70, t_12);  _to_copy_37 = view_70 = t_12 = None
        view_71 = torch.ops.aten.view.default(addmm_2, [1, 11776, 128]);  addmm_2 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(view_71);  view_71 = None
        view_72 = torch.ops.aten.view.default(sigmoid_2, [11776, 128]);  sigmoid_2 = None
        view_73 = torch.ops.aten.view.default(view_72, [1, 11776, 128]);  view_72 = None
        mul_6 = torch.ops.aten.mul.Tensor(_unsafe_view_1, view_73);  _unsafe_view_1 = view_73 = None
        _to_copy_40 = torch.ops.aten._to_copy.default(masked_fill_2, dtype = torch.float32)
        native_layer_norm_default_5 = torch.ops.aten.native_layer_norm.default(_to_copy_40, [128], None, None, 0.1);  _to_copy_40 = None
        getitem_40 = native_layer_norm_default_5[0]
        _to_copy_41 = torch.ops.aten._to_copy.default(arg10_1, dtype = torch.bfloat16);  arg10_1 = None
        _to_copy_42 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        t_13 = torch.ops.aten.t.default(_to_copy_41);  _to_copy_41 = None
        view_74 = torch.ops.aten.view.default(_to_copy_42, [11776, 128]);  _to_copy_42 = None
        mm_10 = torch.ops.aten.mm.default(view_74, t_13);  view_74 = t_13 = None
        view_75 = torch.ops.aten.view.default(mm_10, [1, 11776, 256]);  mm_10 = None
        split_tensor_4 = torch.ops.aten.split.Tensor(view_75, 128, dim = -1);  view_75 = None
        getitem_43 = split_tensor_4[0]
        getitem_44 = split_tensor_4[1];  split_tensor_4 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_43, 1);  getitem_43 = None
        mul_7 = torch.ops.aten.mul.Tensor(getitem_40, add_15);  getitem_40 = add_15 = None
        add_16 = torch.ops.aten.add.Tensor(mul_7, getitem_44);  mul_7 = getitem_44 = None
        _to_copy_43 = torch.ops.aten._to_copy.default(arg11_1, dtype = torch.bfloat16);  arg11_1 = None
        _to_copy_44 = torch.ops.aten._to_copy.default(add_16, dtype = torch.bfloat16);  add_16 = None
        t_14 = torch.ops.aten.t.default(_to_copy_43);  _to_copy_43 = None
        view_76 = torch.ops.aten.view.default(_to_copy_44, [11776, 128]);  _to_copy_44 = None
        mm_11 = torch.ops.aten.mm.default(view_76, t_14);  view_76 = t_14 = None
        view_77 = torch.ops.aten.view.default(mm_11, [1, 11776, 512]);  mm_11 = None
        split_tensor_5 = torch.ops.aten.split.Tensor(view_77, 256, dim = -1);  view_77 = None
        getitem_45 = split_tensor_5[0]
        getitem_46 = split_tensor_5[1];  split_tensor_5 = None
        silu_1 = torch.ops.aten.silu.default(getitem_45);  getitem_45 = None
        mul_8 = torch.ops.aten.mul.Tensor(silu_1, getitem_46);  silu_1 = getitem_46 = None
        _to_copy_45 = torch.ops.aten._to_copy.default(arg14_1, dtype = torch.bfloat16);  arg14_1 = None
        _to_copy_46 = torch.ops.aten._to_copy.default(arg13_1, dtype = torch.bfloat16);  arg13_1 = None
        _to_copy_47 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        view_78 = torch.ops.aten.view.default(_to_copy_47, [11776, 128]);  _to_copy_47 = None
        t_15 = torch.ops.aten.t.default(_to_copy_46);  _to_copy_46 = None
        addmm_3 = torch.ops.aten.addmm.default(_to_copy_45, view_78, t_15);  _to_copy_45 = view_78 = t_15 = None
        view_79 = torch.ops.aten.view.default(addmm_3, [1, 11776, 128]);  addmm_3 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(view_79);  view_79 = None
        _to_copy_48 = torch.ops.aten._to_copy.default(arg12_1, dtype = torch.bfloat16);  arg12_1 = None
        t_16 = torch.ops.aten.t.default(_to_copy_48);  _to_copy_48 = None
        view_80 = torch.ops.aten.view.default(mul_8, [11776, 256]);  mul_8 = None
        mm_12 = torch.ops.aten.mm.default(view_80, t_16);  view_80 = t_16 = None
        view_81 = torch.ops.aten.view.default(mm_12, [1, 11776, 128]);  mm_12 = None
        mul_9 = torch.ops.aten.mul.Tensor(sigmoid_3, view_81);  sigmoid_3 = view_81 = None
        add_17 = torch.ops.aten.add.Tensor(masked_fill_2, mul_9);  masked_fill_2 = mul_9 = None
        add_18 = torch.ops.aten.add.Tensor(add_17, mul_6);  add_17 = mul_6 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(view_22, -1);  view_22 = None
        bitwise_not_4 = torch.ops.aten.bitwise_not.default(unsqueeze_30);  unsqueeze_30 = None
        masked_fill_4 = torch.ops.aten.masked_fill.Scalar(add_18, bitwise_not_4, 0.0);  add_18 = bitwise_not_4 = None
        _to_copy_49 = torch.ops.aten._to_copy.default(getitem_3, dtype = torch.bfloat16);  getitem_3 = None
        _to_copy_50 = torch.ops.aten._to_copy.default(getitem_8, dtype = torch.bfloat16);  getitem_8 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(_to_copy_49, 5);  _to_copy_49 = None
        permute_22 = torch.ops.aten.permute.default(unsqueeze_31, [0, 5, 1, 2, 3, 4]);  unsqueeze_31 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(_to_copy_50, 2);  _to_copy_50 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, 3);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(unsqueeze_33, 4);  unsqueeze_33 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, 5);  unsqueeze_34 = None
        permute_23 = torch.ops.aten.permute.default(unsqueeze_35, [2, 0, 3, 4, 5, 1]);  unsqueeze_35 = None
        permute_24 = torch.ops.aten.permute.default(permute_22, [2, 3, 4, 5, 0, 1]);  permute_22 = None
        view_82 = torch.ops.aten.view.default(permute_24, [1, 1507328, 16]);  permute_24 = None
        permute_25 = torch.ops.aten.permute.default(permute_23, [5, 0, 1, 2, 3, 4]);  permute_23 = None
        view_83 = torch.ops.aten.view.default(permute_25, [1, 16, 4]);  permute_25 = None
        bmm_4 = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84 = torch.ops.aten.view.default(bmm_4, [368, 32, 128, 1, 1, 4]);  bmm_4 = None
        permute_26 = torch.ops.aten.permute.default(view_84, [4, 5, 0, 1, 2, 3]);  view_84 = None
        view_85 = torch.ops.aten.view.default(permute_26, [1, 4, 368, 32, 128]);  permute_26 = None
        view_86 = torch.ops.aten.view.default(view_21, [1, 1, 368, 32, 128]);  view_21 = None
        bitwise_not_5 = torch.ops.aten.bitwise_not.default(view_86);  view_86 = None
        masked_fill_5 = torch.ops.aten.masked_fill.Scalar(view_85, bitwise_not_5, -10000);  view_85 = bitwise_not_5 = None
        _to_copy_51 = torch.ops.aten._to_copy.default(masked_fill_4, dtype = torch.float32)
        native_layer_norm_default_6 = torch.ops.aten.native_layer_norm.default(_to_copy_51, [128], None, None, 0.1);  _to_copy_51 = None
        getitem_47 = native_layer_norm_default_6[0]
        _to_copy_52 = torch.ops.aten._to_copy.default(arg31_1, dtype = torch.bfloat16);  arg31_1 = None
        _to_copy_53 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        t_17 = torch.ops.aten.t.default(_to_copy_52);  _to_copy_52 = None
        view_87 = torch.ops.aten.view.default(_to_copy_53, [11776, 128]);  _to_copy_53 = None
        mm_13 = torch.ops.aten.mm.default(view_87, t_17);  view_87 = t_17 = None
        view_88 = torch.ops.aten.view.default(mm_13, [1, 11776, 256]);  mm_13 = None
        split_tensor_6 = torch.ops.aten.split.Tensor(view_88, 128, dim = -1);  view_88 = None
        getitem_50 = split_tensor_6[0]
        getitem_51 = split_tensor_6[1];  split_tensor_6 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_50, 1);  getitem_50 = None
        mul_10 = torch.ops.aten.mul.Tensor(getitem_47, add_19);  getitem_47 = add_19 = None
        add_20 = torch.ops.aten.add.Tensor(mul_10, getitem_51);  mul_10 = getitem_51 = None
        _to_copy_54 = torch.ops.aten._to_copy.default(add_20, dtype = torch.bfloat16);  add_20 = None
        _to_copy_55 = torch.ops.aten._to_copy.default(arg32_1, dtype = torch.bfloat16);  arg32_1 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(_to_copy_54, 3);  _to_copy_54 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, 4);  unsqueeze_36 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(unsqueeze_37, 5);  unsqueeze_37 = None
        permute_27 = torch.ops.aten.permute.default(unsqueeze_38, [3, 0, 4, 1, 5, 2]);  unsqueeze_38 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(_to_copy_55, 4);  _to_copy_55 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(unsqueeze_39, 5);  unsqueeze_39 = None
        permute_28 = torch.ops.aten.permute.default(unsqueeze_40, [0, 4, 1, 5, 2, 3]);  unsqueeze_40 = None
        permute_29 = torch.ops.aten.permute.default(permute_27, [3, 5, 0, 1, 2, 4]);  permute_27 = None
        view_89 = torch.ops.aten.view.default(permute_29, [1, 11776, 128]);  permute_29 = None
        permute_30 = torch.ops.aten.permute.default(permute_28, [5, 0, 1, 2, 4, 3]);  permute_28 = None
        view_90 = torch.ops.aten.view.default(permute_30, [1, 128, 384]);  permute_30 = None
        bmm_5 = torch.ops.aten.bmm.default(view_89, view_90);  view_89 = view_90 = None
        view_91 = torch.ops.aten.view.default(bmm_5, [11776, 1, 3, 1, 4, 32]);  bmm_5 = None
        permute_31 = torch.ops.aten.permute.default(view_91, [2, 3, 4, 0, 5, 1]);  view_91 = None
        view_92 = torch.ops.aten.view.default(permute_31, [3, 1, 4, 11776, 32]);  permute_31 = None
        view_93 = torch.ops.aten.view.default(view_92, [3, 4, 11776, 32]);  view_92 = None
        unbind_int_3 = torch.ops.aten.unbind.int(view_93);  view_93 = None
        getitem_52 = unbind_int_3[0]
        getitem_53 = unbind_int_3[1]
        getitem_54 = unbind_int_3[2];  unbind_int_3 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(arg30_1, 0);  arg30_1 = None
        expand_9 = torch.ops.aten.expand.default(unsqueeze_41, [1, -1, -1]);  unsqueeze_41 = None
        view_94 = torch.ops.aten.view.default(expand_9, [4, 1, 32]);  expand_9 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_52, view_94);  getitem_52 = view_94 = None
        view_95 = torch.ops.aten.view.default(add_21, [4, 368, 32, 32]);  add_21 = None
        slice_15 = torch.ops.aten.slice.Tensor(getitem_53, dim = 0, start = 0, end = 9223372036854775807);  getitem_53 = None
        slice_16 = torch.ops.aten.slice.Tensor(slice_15, dim = 2, start = 0, end = 9223372036854775807);  slice_15 = None
        index_6 = torch.ops.aten.index.Tensor(slice_16, [None, remainder]);  slice_16 = None
        slice_17 = torch.ops.aten.slice.Tensor(getitem_54, dim = 0, start = 0, end = 9223372036854775807);  getitem_54 = None
        slice_18 = torch.ops.aten.slice.Tensor(slice_17, dim = 2, start = 0, end = 9223372036854775807);  slice_17 = None
        index_7 = torch.ops.aten.index.Tensor(slice_18, [None, remainder]);  slice_18 = remainder = None
        view_96 = torch.ops.aten.view.default(masked_fill_5, [4, 368, 32, 128]);  masked_fill_5 = None
        _to_copy_56 = torch.ops.aten._to_copy.default(view_95, dtype = torch.bfloat16);  view_95 = None
        expand_10 = torch.ops.aten.expand.default(view_96, [4, 368, 32, 128]);  view_96 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(_to_copy_56, index_6, index_7, expand_10, False);  _to_copy_56 = index_6 = index_7 = expand_10 = None
        getitem_55 = _scaled_dot_product_efficient_attention_default_2[0]
        view_97 = torch.ops.aten.view.default(getitem_55, [1, 4, 368, 32, 32]);  getitem_55 = None
        permute_32 = torch.ops.aten.permute.default(view_97, [0, 2, 3, 1, 4]);  view_97 = None
        clone_2 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_2, [1, 11776, 128]);  clone_2 = None
        _to_copy_57 = torch.ops.aten._to_copy.default(arg34_1, dtype = torch.bfloat16);  arg34_1 = None
        _to_copy_58 = torch.ops.aten._to_copy.default(arg33_1, dtype = torch.bfloat16);  arg33_1 = None
        _to_copy_59 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        view_98 = torch.ops.aten.view.default(_to_copy_59, [11776, 128]);  _to_copy_59 = None
        t_18 = torch.ops.aten.t.default(_to_copy_58);  _to_copy_58 = None
        addmm_4 = torch.ops.aten.addmm.default(_to_copy_57, view_98, t_18);  _to_copy_57 = view_98 = t_18 = None
        view_99 = torch.ops.aten.view.default(addmm_4, [1, 11776, 128]);  addmm_4 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(view_99);  view_99 = None
        view_100 = torch.ops.aten.view.default(sigmoid_4, [11776, 128]);  sigmoid_4 = None
        view_101 = torch.ops.aten.view.default(view_100, [1, 11776, 128]);  view_100 = None
        mul_11 = torch.ops.aten.mul.Tensor(_unsafe_view_2, view_101);  _unsafe_view_2 = view_101 = None
        _to_copy_60 = torch.ops.aten._to_copy.default(masked_fill_4, dtype = torch.float32)
        native_layer_norm_default_7 = torch.ops.aten.native_layer_norm.default(_to_copy_60, [128], None, None, 0.1);  _to_copy_60 = None
        getitem_59 = native_layer_norm_default_7[0]
        _to_copy_61 = torch.ops.aten._to_copy.default(arg15_1, dtype = torch.bfloat16);  arg15_1 = None
        _to_copy_62 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16)
        t_19 = torch.ops.aten.t.default(_to_copy_61);  _to_copy_61 = None
        view_102 = torch.ops.aten.view.default(_to_copy_62, [11776, 128]);  _to_copy_62 = None
        mm_14 = torch.ops.aten.mm.default(view_102, t_19);  view_102 = t_19 = None
        view_103 = torch.ops.aten.view.default(mm_14, [1, 11776, 256]);  mm_14 = None
        split_tensor_7 = torch.ops.aten.split.Tensor(view_103, 128, dim = -1);  view_103 = None
        getitem_62 = split_tensor_7[0]
        getitem_63 = split_tensor_7[1];  split_tensor_7 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_62, 1);  getitem_62 = None
        mul_12 = torch.ops.aten.mul.Tensor(getitem_59, add_22);  getitem_59 = add_22 = None
        add_23 = torch.ops.aten.add.Tensor(mul_12, getitem_63);  mul_12 = getitem_63 = None
        _to_copy_63 = torch.ops.aten._to_copy.default(arg16_1, dtype = torch.bfloat16);  arg16_1 = None
        _to_copy_64 = torch.ops.aten._to_copy.default(add_23, dtype = torch.bfloat16);  add_23 = None
        t_20 = torch.ops.aten.t.default(_to_copy_63);  _to_copy_63 = None
        view_104 = torch.ops.aten.view.default(_to_copy_64, [11776, 128]);  _to_copy_64 = None
        mm_15 = torch.ops.aten.mm.default(view_104, t_20);  view_104 = t_20 = None
        view_105 = torch.ops.aten.view.default(mm_15, [1, 11776, 512]);  mm_15 = None
        split_tensor_8 = torch.ops.aten.split.Tensor(view_105, 256, dim = -1);  view_105 = None
        getitem_64 = split_tensor_8[0]
        getitem_65 = split_tensor_8[1];  split_tensor_8 = None
        silu_2 = torch.ops.aten.silu.default(getitem_64);  getitem_64 = None
        mul_13 = torch.ops.aten.mul.Tensor(silu_2, getitem_65);  silu_2 = getitem_65 = None
        _to_copy_65 = torch.ops.aten._to_copy.default(arg19_1, dtype = torch.bfloat16);  arg19_1 = None
        _to_copy_66 = torch.ops.aten._to_copy.default(arg18_1, dtype = torch.bfloat16);  arg18_1 = None
        _to_copy_67 = torch.ops.aten._to_copy.default(view_19, dtype = torch.bfloat16);  view_19 = None
        view_106 = torch.ops.aten.view.default(_to_copy_67, [11776, 128]);  _to_copy_67 = None
        t_21 = torch.ops.aten.t.default(_to_copy_66);  _to_copy_66 = None
        addmm_5 = torch.ops.aten.addmm.default(_to_copy_65, view_106, t_21);  _to_copy_65 = view_106 = t_21 = None
        view_107 = torch.ops.aten.view.default(addmm_5, [1, 11776, 128]);  addmm_5 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(view_107);  view_107 = None
        _to_copy_68 = torch.ops.aten._to_copy.default(arg17_1, dtype = torch.bfloat16);  arg17_1 = None
        t_22 = torch.ops.aten.t.default(_to_copy_68);  _to_copy_68 = None
        view_108 = torch.ops.aten.view.default(mul_13, [11776, 256]);  mul_13 = None
        mm_16 = torch.ops.aten.mm.default(view_108, t_22);  view_108 = t_22 = None
        view_109 = torch.ops.aten.view.default(mm_16, [1, 11776, 128]);  mm_16 = None
        mul_14 = torch.ops.aten.mul.Tensor(sigmoid_5, view_109);  sigmoid_5 = view_109 = None
        add_24 = torch.ops.aten.add.Tensor(masked_fill_4, mul_14);  masked_fill_4 = mul_14 = None
        add_25 = torch.ops.aten.add.Tensor(add_24, mul_11);  add_24 = mul_11 = None
        view_110 = torch.ops.aten.view.default(add_25, [1, 1, 11776, 128]);  add_25 = None
        _to_copy_69 = torch.ops.aten._to_copy.default(arg38_1, dtype = torch.bfloat16);  arg38_1 = None
        t_23 = torch.ops.aten.t.default(_to_copy_69);  _to_copy_69 = None
        view_111 = torch.ops.aten.view.default(view_110, [11776, 128]);  view_110 = None
        mm_17 = torch.ops.aten.mm.default(view_111, t_23);  view_111 = t_23 = None
        view_112 = torch.ops.aten.view.default(mm_17, [1, 1, 11776, 384]);  mm_17 = None
        relu_3 = torch.ops.aten.relu.default(view_112);  view_112 = None
        view_113 = torch.ops.aten.view.default(relu_3, [11776, 384]);  relu_3 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(arg50_1, 1);  arg50_1 = None
        expand_11 = torch.ops.aten.expand.default(unsqueeze_42, [-1, 1, -1]);  unsqueeze_42 = None
        view_116 = torch.ops.aten.view.default(expand_11, [1, 11776]);  expand_11 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(arg51_1, 1);  arg51_1 = None
        expand_12 = torch.ops.aten.expand.default(unsqueeze_43, [-1, 1, -1]);  unsqueeze_43 = None
        view_117 = torch.ops.aten.view.default(expand_12, [1, 11776]);  expand_12 = None
        view_118 = torch.ops.aten.view.default(view_113, [1, 1, 11776, 384]);  view_113 = None
        view_119 = torch.ops.aten.view.default(view_118, [1, 11776, 384]);  view_118 = None
        new_zeros = torch.ops.aten.new_zeros.default(view_119, [1, 512, 384], pin_memory = False)
        new_zeros_1 = torch.ops.aten.new_zeros.default(view_119, [1, 512], pin_memory = False)
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(view_117, 2)
        expand_13 = torch.ops.aten.expand.default(unsqueeze_44, [-1, -1, 384]);  unsqueeze_44 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(view_116, -1)
        mul_15 = torch.ops.aten.mul.Tensor(view_119, unsqueeze_45);  view_119 = unsqueeze_45 = None
        scatter_reduce = torch.ops.aten.scatter_reduce.two(new_zeros, 1, expand_13, mul_15, 'sum');  new_zeros = expand_13 = mul_15 = None
        _to_copy_70 = torch.ops.aten._to_copy.default(view_116, dtype = torch.bfloat16);  view_116 = None
        scatter_reduce_1 = torch.ops.aten.scatter_reduce.two(new_zeros_1, 1, view_117, _to_copy_70, 'sum');  new_zeros_1 = view_117 = _to_copy_70 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(scatter_reduce_1, -1);  scatter_reduce_1 = None
        clamp = torch.ops.aten.clamp.default(unsqueeze_46, min = 1);  unsqueeze_46 = None
        div = torch.ops.aten.div.Tensor(scatter_reduce, clamp);  scatter_reduce = clamp = None
        view_120 = torch.ops.aten.view.default(div, [1, 1, 512, 384]);  div = None
        view_121 = torch.ops.aten.view.default(view_120, [1, 512, 384]);  view_120 = None
        cat = torch.ops.aten.cat.default([view_121, arg43_1], dim = -1);  view_121 = arg43_1 = None
        _to_copy_71 = torch.ops.aten._to_copy.default(arg40_1, dtype = torch.bfloat16);  arg40_1 = None
        t_24 = torch.ops.aten.t.default(_to_copy_71);  _to_copy_71 = None
        view_122 = torch.ops.aten.view.default(cat, [512, 768])
        mm_18 = torch.ops.aten.mm.default(view_122, t_24);  view_122 = t_24 = None
        view_123 = torch.ops.aten.view.default(mm_18, [1, 512, 384]);  mm_18 = None
        _to_copy_72 = torch.ops.aten._to_copy.default(arg39_1, dtype = torch.bfloat16);  arg39_1 = None
        t_25 = torch.ops.aten.t.default(_to_copy_72);  _to_copy_72 = None
        view_124 = torch.ops.aten.view.default(cat, [512, 768]);  cat = None
        mm_19 = torch.ops.aten.mm.default(view_124, t_25);  view_124 = t_25 = None
        view_125 = torch.ops.aten.view.default(mm_19, [1, 512, 384]);  mm_19 = None
        _to_copy_73 = torch.ops.aten._to_copy.default(arg42_1, dtype = torch.bfloat16);  arg42_1 = None
        t_26 = torch.ops.aten.t.default(_to_copy_73);  _to_copy_73 = None
        view_126 = torch.ops.aten.view.default(view_125, [512, 384])
        mm_20 = torch.ops.aten.mm.default(view_126, t_26);  view_126 = t_26 = None
        view_127 = torch.ops.aten.view.default(mm_20, [1, 512, 512]);  mm_20 = None
        split_tensor_9 = torch.ops.aten.split.Tensor(view_127, 256, dim = -1);  view_127 = None
        getitem_66 = split_tensor_9[0]
        getitem_67 = split_tensor_9[1];  split_tensor_9 = None
        view_128 = torch.ops.aten.view.default(getitem_66, [1, 512, 1, 256]);  getitem_66 = None
        add_26 = torch.ops.aten.add.Tensor(arg44_1, view_128);  arg44_1 = view_128 = None
        view_129 = torch.ops.aten.view.default(getitem_67, [1, 1, 512, 256]);  getitem_67 = None
        add_27 = torch.ops.aten.add.Tensor(add_26, view_129);  add_26 = view_129 = None
        _to_copy_74 = torch.ops.aten._to_copy.default(arg41_1, dtype = torch.bfloat16);  arg41_1 = None
        t_27 = torch.ops.aten.t.default(_to_copy_74);  _to_copy_74 = None
        view_130 = torch.ops.aten.view.default(add_27, [262144, 256]);  add_27 = None
        mm_21 = torch.ops.aten.mm.default(view_130, t_27);  view_130 = t_27 = None
        view_131 = torch.ops.aten.view.default(mm_21, [1, 512, 512, 256]);  mm_21 = None
        return (view_125, view_123, view_131)
        
    # To see more debug info, please use `graph_module.print_readable()`