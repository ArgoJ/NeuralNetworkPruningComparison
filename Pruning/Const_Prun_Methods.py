from Pruning.PruningMethods.LotteryTicket import lotteryTicket
from Pruning.PruningMethods.Finetune import finetune
from Pruning.PruningMethods.SynFlow import synFlow_and_train



PRUN_METHODS = {
    'lottery_ticket': lotteryTicket,
    'finetune': finetune,
    'syn_flow': synFlow_and_train
}
