import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.io import loadmat
from tqdm import tqdm
from utils import pearson_r


HCP_CONTRASTS=['01_EMOTION_FACES','02_EMOTION_SHAPES','03_EMOTION_FACES-SHAPES',
           '07_GAMBLING_PUNISH','08_GAMBLING_REWARD','09_GAMBLING_PUNISH-REWARD',
           '13_LANGUAGE_MATH','14_LANGUAGE_STORY','15_LANGUAGE_MATH-STORY',
           '19_MOTOR_CUE','20_MOTOR_LF','21_MOTOR_LH','22_MOTOR_RF','23_MOTOR_RH','24_MOTOR_T',
           '25_MOTOR_AVG','26_MOTOR_CUE-AVG','27_MOTOR_LF-AVG','28_MOTOR_LH-AVG','29_MOTOR_RF-AVG',
           '30_MOTOR_RH-AVG','31_MOTOR_T-AVG','45_RELATIONAL_MATCH','46_RELATIONAL_REL',
           '47_RELATIONAL_MATCH-REL','51_SOCIAL_RANDOM','52_SOCIAL_TOM','53_SOCIAL_RANDOM-TOM',
           '57_WM_2BK_BODY','58_WM_2BK_FACE','59_WM_2BK_PLACE','60_WM_2BK_TOOL','61_WM_0BK_BODY',
           '62_WM_0BK_FACE','63_WM_0BK_PLACE','64_WM_0BK_TOOL','65_WM_2BK','66_WM_0BK','67_WM_2BK-0BK',
           '71_WM_BODY','72_WM_FACE','73_WM_PLACE','74_WM_TOOL','75_WM_BODY-AVG','76_WM_FACE-AVG','77_WM_PLACE-AVG',
           '78_WM_TOOL-AVG']

UKB_CONTRASTS = ['EMOTION_FACES','EMOTION_SHAPES','EMOTION_FACES-SHAPES']


