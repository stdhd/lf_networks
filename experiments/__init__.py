from experiments.hgn_combined import HgnTrainer
from experiments.FirstStageExperiment import FirstStageTrainer
from experiments.gru_baseline import GruBaselineTrainer
from experiments.SecondStageExperiment import SecondStageTrainer
from experiments.TamingExperiment import TamingTrainer
from experiments.MotionTamingExperiment import MotionTamingTrainer
from experiments.FirstStageExperimentGanLoss import FirstStageTrainerGAN

__experiments__ = {
    'hgncombined': HgnTrainer,
    'stochasticmotion': FirstStageTrainer,
    'gru_baseline': GruBaselineTrainer,
    'secondstage': SecondStageTrainer,
    'firststage': FirstStageTrainer,
    'taming': TamingTrainer,
    'motion_taming': MotionTamingTrainer,
    'firststagegan': FirstStageTrainerGAN,


}


def select_experiment(params, resume=False):
    experiment = params['experiment']
    return __experiments__[experiment]#(params=params, resume=resume)