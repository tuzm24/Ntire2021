class LearningIndex:
    TEST = 0
    TRAINING = 1
    VALIDATION = 2
    MAX_NUM_COMPONENT = 3
    INDEX_DIC = {TEST:"TEST", TRAINING:"TRAINING", VALIDATION:"VALIDATION"}

class Component:
    COMPONENT_Y = 0
    COMPONENT_Cb = 1
    COMPONENT_Cr = 2
    MAX_NUM_COMPONENT = 3
    INDEX_DIC = {COMPONENT_Y: 'Y', COMPONENT_Cb: 'Cb', COMPONENT_Cr: 'Cr'}


class ChromaFormat:
    YCbCr4_0_0 = 0
    YCbCr4_2_0 = 1
    YCbCr4_4_4 = 2
    MAX_NUM_COMPONENT = 3

class ChannelType:
    CHANNEL_TYPE_LUMA = 0
    CHANNEL_TYPE_CHROMA = 1
    MAX_NUM_CHANNEL_TYPE = 2
    STR_LUMA = 'LUMA'
    STR_CHROMA = 'CHROMA'

class PictureFormat:
    ORIGINAL = 0
    PREDICTION = 1
    RECONSTRUCTION = 2
    UNFILTEREDRECON = 3
    MAX_NUM_COMPONENT = 4
    INDEX_DIC = {ORIGINAL:'ORIGINAL', PREDICTION:'PREDICTION',
                 RECONSTRUCTION:'RECONSTRUCTION',
                 UNFILTEREDRECON:'UNFILTEREDRECON'}

class UnitFormat:
    CU = 0
    PU = 1
    TU = 2
    MAX_NUM_COMPONENT = 3
    INDEX_DIC = {CU:'CU', PU:'PU', TU:'TU'}

class BlockType:
    PredMode = 'PredMode'
    QT_Depth = 'QT_Depth'
    BT_Depth = 'BT_Depth'
    MT_Depth = 'MT_Depth'
    ChromaQPAdj = 'ChromaQPAdj'
    QP = 'QP'
    SplitSeries = 'SplitSeries'
    MTS_Y = 'MTS_Y'
    MTS_Cb = 'MTS_Cb'
    MTS_Cr = 'MTS_Cr'
    BDPCM = 'BDPCM'
    BDPCMChroma = 'BDPCMChroma'
    TileIdx = 'TileIdx'
    IndependentSliceIdx = 'IndependentSliceIdx'
    LFNSTIdx = 'LFNSTIdx'
    JointCbCr = 'JointCbCr'
    CompAlphaCb = 'CompAlphaCb'
    CompAlphaCr = 'CompAlphaCr'
    RDPCM_Y = 'RDPCM_Y'
    RDPCM_Cb = 'RDPCM_Cb'
    RDPCM_Cr = 'RDPCM_Cr'
    Luma_IntraMode = 'Luma_IntraMode'
    Chroma_IntraMode = 'Chroma_IntraMode'
    MultiRefIdx = 'MultiRefIdx'
    MIPFlag = 'MIPFlag'
    ISPMode = 'ISPMode'
    SkipFlag = 'SkipFlag'
    RootCbf = 'RootCbf'
    SbtIdx = 'SbtIdx'
    SbtPos = 'SbtPos'
    Cbf_Y = 'Cbf_Y'
    Cbf_Cb = 'Cbf_Cb'
    Cbf_Cr = 'Cbf_Cr'
    IMVMode = 'IMVMode'
    InterDir = 'InterDir'
    MergeFlag = 'MergeFlag'
    RegularMergeFlag = 'RegularMergeFlag'
    MergeIdx = 'MergeIdx'
    MergeType = 'MergeType'
    MVPIdxL0 = 'MVPIdxL0'
    MVPIdxL1 = 'MVPIdxL1'
    MVL0 = 'MVL0'
    MVL1 = 'MVL1'
    MVDL0 = 'MVDL0'
    MVDL1 = 'MVDL1'
    MotionBufL0 = 'MotionBufL0'
    MotionBufL1 = 'MotionBufL1'
    RefIdxL0 = 'RefIdxL0'
    RefIdxL1 = 'RefIdxL1'
    AffineFlag = 'AffineFlag'
    AffineMVL0 = 'AffineMVL0'
    AffineMVL1 = 'AffineMVL1'
    AffineType = 'AffineType'
    MMVDSkipFlag = 'MMVDSkipFlag'
    MMVDMergeFlag = 'MMVDMergeFlag'
    MMVDMergeIdx = 'MMVDMergeIdx'
    CiipFlag = 'CiipFlag'
    SMVDFlag = 'SMVDFlag'
    GeoPartitioning = 'GeoPartitioning'
    GeoMVL0 = 'GeoMVL0'
    GeoMVL1 = 'GeoMVL1'
    BCWIndex = 'BCWIndex'
    Depth_Chroma = 'Depth_Chroma'
    QT_Depth_Chroma = 'QT_Depth_Chroma'
    BT_Depth_Chroma = 'BT_Depth_Chroma'
    MT_Depth_Chroma = 'MT_Depth_Chroma'
    ChromaQPAdj_Chroma = 'ChromaQPAdj_Chroma'
    QP_Chroma = 'QP_Chroma'
    SplitSeries_Chroma = 'SplitSeries_Chroma'


ChromaScale = 2


class PPSType:
    QP = 'QP'
    SliceType = 'SliceType'
    LayerID = 'LayerID'
    TemporalID = 'TemporalID'
    L0 = 'L0'
    L1 = 'L1'