import torch
import torch.nn as nn

class feature_embedding_module(nn.Module):
    def  __init__(self, model):
        super(feature_embedding_module, self).__init__()
        self.device = torch.device('cuda')
        self.arg0_1 = model.feature_embeddings.TEMPLATES.TemplateResType.embedding.weight
        self.arg1_1 = getattr(model.input_projs.ATOM, "0").weight
        self.arg2_1 = getattr(model.input_projs.ATOM, "0").bias
        self.arg3_1 = getattr(model.input_projs.ATOM_PAIR, "0").weight
        self.arg4_1 = getattr(model.input_projs.ATOM_PAIR, "0").bias
        self.arg5_1 = getattr(model.input_projs.TOKEN, "0").weight
        self.arg6_1 = getattr(model.input_projs.TOKEN, "0").bias
        self.arg7_1 = getattr(model.input_projs.TOKEN_PAIR, "0").weight
        self.arg8_1 = getattr(model.input_projs.TOKEN_PAIR, "0").bias
        self.arg9_1 = getattr(model.input_projs.MSA, "0").weight
        self.arg10_1 = getattr(model.input_projs.MSA, "0").bias
        self.arg11_1 = getattr(model.input_projs.TEMPLATES, "0").weight
        self.arg12_1 = getattr(model.input_projs.TEMPLATES, "0").bias
        arg13_1 = model.feature_embeddings.TOKEN_PAIR.TokenDistanceRestraint.radii
        self.register_buffer('arg13_1', arg13_1)
        arg14_1 = model.feature_embeddings.TOKEN_PAIR.TokenPairPocketRestraint.radii
        self.register_buffer('arg14_1', arg14_1)
        # arg15_1 = model.feature_embeddings.TEMPLATES.TemplateResType.offsets
        # self.register_buffer('arg15_1', arg15_1)
        self._lifted_tensor_constant0 = model._lifted_tensor_constant0
        self._lifted_tensor_constant1 = model._lifted_tensor_constant1

    
    # def forward(self, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1):
    def forward(self, TemplateDistogram, TemplateMask, TemplateResType, TemplateUnitVector,
            ChainIsCropped, ESMEmbeddings, IsDistillation, MSADeletionMean, MSAProfile,
            MissingChainContact, ResidueType, TokenBFactor, TokenPLDDT,
            BlockedAtomPairDistogram, InverseSquaredBlockedAtomPairDistances, IsPairedMSA,
            MSADataSource, MSADeletionValue, MSAHasDeletion, MSAOneHot,
            DockingConstraintGenerator, RelativeChain, RelativeEntity,
            RelativeSequenceSeparation, RelativeTokenSeparation,
            TokenDistanceRestraint, TokenPairPocketRestraint,
            AtomNameOneHot, AtomRefCharge, AtomRefElement, AtomRefMask, AtomRefPos):
        arg16_1 = TemplateDistogram
        arg17_1 = TemplateMask
        arg18_1 = TemplateResType
        arg19_1 = TemplateUnitVector
        arg20_1 = ChainIsCropped
        arg21_1 = ESMEmbeddings
        arg22_1 = IsDistillation
        arg23_1 = MSADeletionMean
        arg24_1 = MSAProfile
        arg25_1 = MissingChainContact
        arg26_1 = ResidueType
        arg27_1 = TokenBFactor
        arg28_1 = TokenPLDDT
        arg29_1 = BlockedAtomPairDistogram
        arg30_1 = InverseSquaredBlockedAtomPairDistances
        arg31_1 = IsPairedMSA
        arg32_1 = MSADataSource
        arg33_1 = MSADeletionValue
        arg34_1 = MSAHasDeletion
        arg35_1 = MSAOneHot
        arg36_1 = DockingConstraintGenerator
        arg37_1 = RelativeChain
        arg38_1 = RelativeEntity
        arg39_1 = RelativeSequenceSeparation
        arg40_1 = RelativeTokenSeparation
        arg41_1 = TokenDistanceRestraint
        arg42_1 = TokenPairPocketRestraint
        arg43_1 = AtomNameOneHot
        arg44_1 = AtomRefCharge
        arg45_1 = AtomRefElement
        arg46_1 = AtomRefMask
        arg47_1 = AtomRefPos
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
        # arg15_1 = self.arg15_1
        _lifted_tensor_constant0 = self._lifted_tensor_constant0
        _lifted_tensor_constant1 = self._lifted_tensor_constant1

        _to_copy = arg43_1.to(dtype = torch.int64) ;  arg43_1 = None
        arange = torch.arange(65,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze = torch.unsqueeze(_to_copy,-1) ;  _to_copy = None
        eq = (unsqueeze == arange) ;  unsqueeze = arange = None
        _to_copy_1 = eq.to(dtype = torch.int64) ;  eq = None
        _to_copy_2 = _to_copy_1.to(dtype = torch.float32) ;  _to_copy_1 = None
        view = _to_copy_2.view(1, 11776, 260) ;  _to_copy_2 = None
        view_1 = arg44_1.view(1, 11776, 1) ;  arg44_1 = None
        _to_copy_3 = arg45_1.to(dtype = torch.int64) ;  arg45_1 = None
        arange_1 = torch.arange(130,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_1 = torch.unsqueeze(_to_copy_3,-1) ;  _to_copy_3 = None
        eq_1 = (unsqueeze_1 == arange_1) ;  unsqueeze_1 = arange_1 = None
        _to_copy_4 = eq_1.to(dtype = torch.int64) ;  eq_1 = None
        _to_copy_5 = _to_copy_4.to(dtype = torch.float32) ;  _to_copy_4 = None
        view_2 = _to_copy_5.view(1, 11776, 130) ;  _to_copy_5 = None
        view_3 = arg46_1.view(1, 11776, 1) ;  arg46_1 = None
        view_4 = arg47_1.view(1, 11776, 3) ;  arg47_1 = None
        cat = torch.cat([view,view_1,view_2,view_3,view_4],dim = -1) ;  view = view_1 = view_2 = view_3 = view_4 = None
        _to_copy_6 = arg2_1.to(dtype = torch.bfloat16) ;  arg2_1 = None
        _to_copy_7 = arg1_1.to(dtype = torch.bfloat16) ;  arg1_1 = None
        _to_copy_8 = cat.to(dtype = torch.bfloat16) ;  cat = None
        view_5 = _to_copy_8.view(11776, 395) ;  _to_copy_8 = None
        t = _to_copy_7.t() ;  _to_copy_7 = None
        addmm = torch.addmm(_to_copy_6,view_5,t) ;  _to_copy_6 = view_5 = t = None
        view_6 = addmm.view(1, 11776, 256) ;  addmm = None
        _to_copy_9 = arg29_1.to(dtype = torch.int64) ;  arg29_1 = None
        arange_2 = torch.arange(12,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_2 = torch.unsqueeze(_to_copy_9,-1) ;  _to_copy_9 = None
        eq_2 = (unsqueeze_2 == arange_2) ;  unsqueeze_2 = arange_2 = None
        _to_copy_10 = eq_2.to(dtype = torch.int64) ;  eq_2 = None
        _to_copy_11 = _to_copy_10.to(dtype = torch.float32) ;  _to_copy_10 = None
        view_7 = _to_copy_11.view(1, 368, 32, 128, 12) ;  _to_copy_11 = None
        view_8 = arg30_1.view(1, 368, 32, 128, 2) ;  arg30_1 = None
        cat_1 = torch.cat([view_7,view_8],dim = -1) ;  view_7 = view_8 = None
        _to_copy_12 = arg4_1.to(dtype = torch.bfloat16) ;  arg4_1 = None
        _to_copy_13 = arg3_1.to(dtype = torch.bfloat16) ;  arg3_1 = None
        _to_copy_14 = cat_1.to(dtype = torch.bfloat16) ;  cat_1 = None
        view_9 = _to_copy_14.view(1507328, 14) ;  _to_copy_14 = None
        t_1 = _to_copy_13.t() ;  _to_copy_13 = None
        addmm_1 = torch.addmm(_to_copy_12,view_9,t_1) ;  _to_copy_12 = view_9 = t_1 = None
        view_10 = addmm_1.view(1, 368, 32, 128, 32) ;  addmm_1 = None
        view_11 = arg20_1.view(1, 512, 1) ;  arg20_1 = None
        view_12 = arg21_1.view(1, 512, 2560) ;  arg21_1 = None
        _to_copy_15 = arg22_1.to(dtype = torch.int64) ;  arg22_1 = None
        arange_3 = torch.arange(2,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_3 = torch.unsqueeze(_to_copy_15,-1) ;  _to_copy_15 = None
        eq_3 = (unsqueeze_3 == arange_3) ;  unsqueeze_3 = arange_3 = None
        _to_copy_16 = eq_3.to(dtype = torch.int64) ;  eq_3 = None
        _to_copy_17 = _to_copy_16.to(dtype = torch.float32) ;  _to_copy_16 = None
        view_13 = _to_copy_17.view(1, 512, 2) ;  _to_copy_17 = None
        view_14 = arg23_1.view(1, 512, 1) ;  arg23_1 = None
        view_15 = arg24_1.view(1, 512, 33) ;  arg24_1 = None
        view_16 = arg25_1.view(1, 512, 1) ;  arg25_1 = None
        _to_copy_18 = arg26_1.to(dtype = torch.int64) ;  arg26_1 = None
        arange_4 = torch.arange(33,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_4 = torch.unsqueeze(_to_copy_18,-1) ;  _to_copy_18 = None
        eq_4 = (unsqueeze_4 == arange_4) ;  unsqueeze_4 = arange_4 = None
        _to_copy_19 = eq_4.to(dtype = torch.int64) ;  eq_4 = None
        _to_copy_20 = _to_copy_19.to(dtype = torch.float32) ;  _to_copy_19 = None
        view_17 = _to_copy_20.view(1, 512, 33) ;  _to_copy_20 = None
        _to_copy_21 = arg27_1.to(dtype = torch.int64) ;  arg27_1 = None
        arange_5 = torch.arange(3,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_5 = torch.unsqueeze(_to_copy_21,-1) ;  _to_copy_21 = None
        eq_5 = (unsqueeze_5 == arange_5) ;  unsqueeze_5 = arange_5 = None
        _to_copy_22 = eq_5.to(dtype = torch.int64) ;  eq_5 = None
        _to_copy_23 = _to_copy_22.to(dtype = torch.float32) ;  _to_copy_22 = None
        view_18 = _to_copy_23.view(1, 512, 3) ;  _to_copy_23 = None
        _to_copy_24 = arg28_1.to(dtype = torch.int64) ;  arg28_1 = None
        arange_6 = torch.arange(4,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_6 = torch.unsqueeze(_to_copy_24,-1) ;  _to_copy_24 = None
        eq_6 = (unsqueeze_6 == arange_6) ;  unsqueeze_6 = arange_6 = None
        _to_copy_25 = eq_6.to(dtype = torch.int64) ;  eq_6 = None
        _to_copy_26 = _to_copy_25.to(dtype = torch.float32) ;  _to_copy_25 = None
        view_19 = _to_copy_26.view(1, 512, 4) ;  _to_copy_26 = None
        cat_2 = torch.cat([view_11,view_12,view_13,view_14,view_15,view_16,view_17,view_18,view_19],dim = -1) ;  view_11 = view_12 = view_13 = view_14 = view_15 = view_16 = view_17 = view_18 = view_19 = None
        _to_copy_27 = arg6_1.to(dtype = torch.bfloat16) ;  arg6_1 = None
        _to_copy_28 = arg5_1.to(dtype = torch.bfloat16) ;  arg5_1 = None
        _to_copy_29 = cat_2.to(dtype = torch.bfloat16) ;  cat_2 = None
        view_20 = _to_copy_29.view(512, 2638) ;  _to_copy_29 = None
        t_2 = _to_copy_28.t() ;  _to_copy_28 = None
        addmm_2 = torch.addmm(_to_copy_27,view_20,t_2) ;  _to_copy_27 = view_20 = t_2 = None
        view_21 = addmm_2.view(1, 512, 384) ;  addmm_2 = None
        _to_copy_30 = arg36_1.to(dtype = torch.int64) ;  arg36_1 = None
        arange_7 = torch.arange(6,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_7 = torch.unsqueeze(_to_copy_30,-1) ;  _to_copy_30 = None
        eq_7 = (unsqueeze_7 == arange_7) ;  unsqueeze_7 = arange_7 = None
        _to_copy_31 = eq_7.to(dtype = torch.int64) ;  eq_7 = None
        _to_copy_32 = _to_copy_31.to(dtype = torch.float32) ;  _to_copy_31 = None
        view_22 = _to_copy_32.view(1, 512, 512, 6) ;  _to_copy_32 = None
        _to_copy_33 = arg37_1.to(dtype = torch.int64) ;  arg37_1 = None
        arange_8 = torch.arange(6,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_8 = torch.unsqueeze(_to_copy_33,-1) ;  _to_copy_33 = None
        eq_8 = (unsqueeze_8 == arange_8) ;  unsqueeze_8 = arange_8 = None
        _to_copy_34 = eq_8.to(dtype = torch.int64) ;  eq_8 = None
        _to_copy_35 = _to_copy_34.to(dtype = torch.float32) ;  _to_copy_34 = None
        view_23 = _to_copy_35.view(1, 512, 512, 6) ;  _to_copy_35 = None
        _to_copy_36 = arg38_1.to(dtype = torch.int64) ;  arg38_1 = None
        arange_9 = torch.arange(3,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_9 = torch.unsqueeze(_to_copy_36,-1) ;  _to_copy_36 = None
        eq_9 = (unsqueeze_9 == arange_9) ;  unsqueeze_9 = arange_9 = None
        _to_copy_37 = eq_9.to(dtype = torch.int64) ;  eq_9 = None
        _to_copy_38 = _to_copy_37.to(dtype = torch.float32) ;  _to_copy_37 = None
        view_24 = _to_copy_38.view(1, 512, 512, 3) ;  _to_copy_38 = None
        _to_copy_39 = arg39_1.to(dtype = torch.int64) ;  arg39_1 = None
        arange_10 = torch.arange(67,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_10 = torch.unsqueeze(_to_copy_39,-1) ;  _to_copy_39 = None
        eq_10 = (unsqueeze_10 == arange_10) ;  unsqueeze_10 = arange_10 = None
        _to_copy_40 = eq_10.to(dtype = torch.int64) ;  eq_10 = None
        _to_copy_41 = _to_copy_40.to(dtype = torch.float32) ;  _to_copy_40 = None
        view_25 = _to_copy_41.view(1, 512, 512, 67) ;  _to_copy_41 = None
        _to_copy_42 = arg40_1.to(dtype = torch.int64) ;  arg40_1 = None
        arange_11 = torch.arange(67,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_11 = torch.unsqueeze(_to_copy_42,-1) ;  _to_copy_42 = None
        eq_11 = (unsqueeze_11 == arange_11) ;  unsqueeze_11 = arange_11 = None
        _to_copy_43 = eq_11.to(dtype = torch.int64) ;  eq_11 = None
        _to_copy_44 = _to_copy_43.to(dtype = torch.float32) ;  _to_copy_43 = None
        view_26 = _to_copy_44.view(1, 512, 512, 67) ;  _to_copy_44 = None
        unsqueeze_12 = torch.unsqueeze(arg41_1,-1) ;  arg41_1 = None
        unsqueeze_13 = torch.unsqueeze(arg13_1,0) ;  arg13_1 = None
        unsqueeze_14 = torch.unsqueeze(unsqueeze_13,1) ;  unsqueeze_13 = None
        unsqueeze_15 = torch.unsqueeze(unsqueeze_14,2) ;  unsqueeze_14 = None
        sub = torch.sub(unsqueeze_15,unsqueeze_12) ;  unsqueeze_15 = None
        div = torch.div(sub,4.800000190734863) ;  sub = None
        pow_1 = torch.pow(div,2) ;  div = None
        clamp_max = torch.clamp_max(pow_1,16) ;  pow_1 = None
        neg = torch.neg(clamp_max) 
        exp = torch.exp(neg) ;  neg = None
        eq_12 = torch.eq(clamp_max,16) ;  clamp_max = None
        lift_fresh_copy = torch.clone(_lifted_tensor_constant0) ;  _lifted_tensor_constant0 = None
        index_put = torch.index_put(exp,[eq_12],lift_fresh_copy) ;  exp = eq_12 = lift_fresh_copy = None
        eq_13 = torch.eq(unsqueeze_12,-1.0) ;  unsqueeze_12 = None
        _to_copy_45 = eq_13.to(dtype = torch.float32) ;  eq_13 = None
        rsub = torch.rsub(_to_copy_45,1) 
        mul = torch.mul(index_put,rsub) ;  index_put = rsub = None
        cat_3 = torch.cat([mul,_to_copy_45],dim = -1) ;  mul = _to_copy_45 = None
        view_27 = cat_3.view(1, 512, 512, 7) ;  cat_3 = None
        unsqueeze_16 = torch.unsqueeze(arg42_1,-1) ;  arg42_1 = None
        unsqueeze_17 = torch.unsqueeze(arg14_1,0) ;  arg14_1 = None
        unsqueeze_18 = torch.unsqueeze(unsqueeze_17,1) ;  unsqueeze_17 = None
        unsqueeze_19 = torch.unsqueeze(unsqueeze_18,2) ;  unsqueeze_18 = None
        sub_1 = torch.sub(unsqueeze_19,unsqueeze_16) ;  unsqueeze_19 = None
        div_1 = torch.div(sub_1,2.799999952316284) ;  sub_1 = None
        pow_2 = torch.pow(div_1,2) ;  div_1 = None
        clamp_max_1 = torch.clamp_max(pow_2,16) ;  pow_2 = None
        neg_1 = torch.neg(clamp_max_1) 
        exp_1 = torch.exp(neg_1) ;  neg_1 = None
        eq_14 = torch.eq(clamp_max_1,16) ;  clamp_max_1 = None
        lift_fresh_copy_1 = torch.clone(_lifted_tensor_constant1) ;  _lifted_tensor_constant1 = None
        index_put_1 = torch.index_put(exp_1,[eq_14],lift_fresh_copy_1) ;  exp_1 = eq_14 = lift_fresh_copy_1 = None
        eq_15 = torch.eq(unsqueeze_16,-1.0) ;  unsqueeze_16 = None
        _to_copy_46 = eq_15.to(dtype = torch.float32) ;  eq_15 = None
        rsub_1 = torch.rsub(_to_copy_46,1) 
        mul_1 = torch.mul(index_put_1,rsub_1) ;  index_put_1 = rsub_1 = None
        cat_4 = torch.cat([mul_1,_to_copy_46],dim = -1) ;  mul_1 = _to_copy_46 = None
        view_28 = cat_4.view(1, 512, 512, 7) ;  cat_4 = None
        cat_5 = torch.cat([view_22,view_23,view_24,view_25,view_26,view_27,view_28],dim = -1) ;  view_22 = view_23 = view_24 = view_25 = view_26 = view_27 = view_28 = None
        _to_copy_47 = arg8_1.to(dtype = torch.bfloat16) ;  arg8_1 = None
        _to_copy_48 = arg7_1.to(dtype = torch.bfloat16) ;  arg7_1 = None
        _to_copy_49 = cat_5.to(dtype = torch.bfloat16) ;  cat_5 = None
        view_29 = _to_copy_49.view(262144, 163) ;  _to_copy_49 = None
        t_3 = _to_copy_48.t() ;  _to_copy_48 = None
        addmm_3 = torch.addmm(_to_copy_47,view_29,t_3) ;  _to_copy_47 = view_29 = t_3 = None
        view_30 = addmm_3.view(1, 512, 512, 512) ;  addmm_3 = None
        view_31 = arg31_1.view(1, 16384, 512, 1) ;  arg31_1 = None
        _to_copy_50 = arg32_1.to(dtype = torch.int64) ;  arg32_1 = None
        arange_12 = torch.arange(6,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_20 = torch.unsqueeze(_to_copy_50,-1) ;  _to_copy_50 = None
        eq_16 = (unsqueeze_20 == arange_12) ;  unsqueeze_20 = arange_12 = None
        _to_copy_51 = eq_16.to(dtype = torch.int64) ;  eq_16 = None
        _to_copy_52 = _to_copy_51.to(dtype = torch.float32) ;  _to_copy_51 = None
        view_32 = _to_copy_52.view(1, 16384, 512, 6) ;  _to_copy_52 = None
        view_33 = arg33_1.view(1, 16384, 512, 1) ;  arg33_1 = None
        view_34 = arg34_1.view(1, 16384, 512, 1) ;  arg34_1 = None
        _to_copy_53 = arg35_1.to(dtype = torch.int64) ;  arg35_1 = None
        arange_13 = torch.arange(33,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_21 = torch.unsqueeze(_to_copy_53,-1) ;  _to_copy_53 = None
        eq_17 = (unsqueeze_21 == arange_13) ;  unsqueeze_21 = arange_13 = None
        _to_copy_54 = eq_17.to(dtype = torch.int64) ;  eq_17 = None
        _to_copy_55 = _to_copy_54.to(dtype = torch.float32) ;  _to_copy_54 = None
        view_35 = _to_copy_55.view(1, 16384, 512, 33) ;  _to_copy_55 = None
        cat_6 = torch.cat([view_31,view_32,view_33,view_34,view_35],dim = -1) ;  view_31 = view_32 = view_33 = view_34 = view_35 = None
        _to_copy_56 = arg10_1.to(dtype = torch.bfloat16) ;  arg10_1 = None
        _to_copy_57 = arg9_1.to(dtype = torch.bfloat16) ;  arg9_1 = None
        _to_copy_58 = cat_6.to(dtype = torch.bfloat16) ;  cat_6 = None
        view_36 = _to_copy_58.view(8388608, 42) ;  _to_copy_58 = None
        t_4 = _to_copy_57.t() ;  _to_copy_57 = None
        addmm_4 = torch.addmm(_to_copy_56,view_36,t_4) ;  _to_copy_56 = view_36 = t_4 = None
        view_37 = addmm_4.view(1, 16384, 512, 64) ;  addmm_4 = None
        _to_copy_59 = arg16_1.to(dtype = torch.int64) ;  arg16_1 = None
        arange_14 = torch.arange(39,dtype = torch.int64,layout = torch.strided,device = self.device) 
        unsqueeze_22 = torch.unsqueeze(_to_copy_59,-1) ;  _to_copy_59 = None
        eq_18 = (unsqueeze_22 == arange_14) ;  unsqueeze_22 = arange_14 = None
        _to_copy_60 = eq_18.to(dtype = torch.int64) ;  eq_18 = None
        _to_copy_61 = _to_copy_60.to(dtype = torch.float32) ;  _to_copy_60 = None
        view_38 = _to_copy_61.view(1, 4, 512, 512, 39) ;  _to_copy_61 = None
        view_39 = arg17_1.view(1, 4, 512, 512, 2) ;  arg17_1 = None
        _to_copy_62 = arg18_1.to(dtype = torch.int64) ;  arg18_1 = None
        arange_15 = torch.arange(1,device = self.device,pin_memory = False) 
        mul_2 = torch.mul(arange_15,33) ;  arange_15 = None
        add = torch.add(_to_copy_62,mul_2) ;  _to_copy_62 = mul_2 = None
        embedding = torch.nn.functional.embedding(add,arg0_1) ;  arg0_1 = add = None
        view_40 = embedding.view(1, 4, 512, 1, 32) 
        view_41 = embedding.view(1, 4, 1, 512, 32) ;  embedding = None
        add_1 = torch.add(view_40,view_41) ;  view_40 = view_41 = None
        view_42 = add_1.view(1, 4, 512, 512, 32) ;  add_1 = None
        view_43 = arg19_1.view(1, 4, 512, 512, 3) ;  arg19_1 = None
        cat_7 = torch.cat([view_38,view_39,view_42,view_43],dim = -1) ;  view_38 = view_39 = view_42 = view_43 = None
        _to_copy_63 = arg12_1.to(dtype = torch.bfloat16) ;  arg12_1 = None
        _to_copy_64 = arg11_1.to(dtype = torch.bfloat16) ;  arg11_1 = None
        _to_copy_65 = cat_7.to(dtype = torch.bfloat16) ;  cat_7 = None
        view_44 = _to_copy_65.view(1048576, 76) ;  _to_copy_65 = None
        t_5 = _to_copy_64.t() ;  _to_copy_64 = None
        addmm_5 = torch.addmm(_to_copy_63,view_44,t_5) ;  _to_copy_63 = view_44 = t_5 = None
        view_45 = addmm_5.view(1, 4, 512, 512, 64) ;  addmm_5 = None
        return (view_6, view_10, view_21, view_30, view_37, view_45)
        
    # To see more debug info, please use `graph_module.print_readable()`