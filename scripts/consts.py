from enum import Enum


class indiciesList(Enum):
    STRENGTH = 'strength'
    DENSITY = 'density'
    CPL = "characteristic_path_length"
    GLOBAL_EFFICIENCY = "global_efficiency"
    ADEGDEG = "adegdeg"
    AODID = "aodid"
    AIDOD = "aidod"
    AIDID = "aidid"
    JOD = "jod"
    JID = "jid"
    JBL = "jbl"
    DFP = "DFP"
    IFP = "IFP"
    DLR = "DLR"
    ILR = "ILR"


class plvTresholdAnova2Data:
    FACTOR_IDX = 7
