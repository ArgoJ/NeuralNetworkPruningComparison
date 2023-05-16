import matplotlib.pyplot as plt

from time import perf_counter


from Saves import Saves
from Loads import loadConfigWithDim
from MultiNets import multiNets
from Helpers import selectDevice, set_multi_thread, del_multi_thread
from DataClassesJSON import ConfigData

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from PlotScripts.PlotHelpers import make_latex_fonts
from PlotScripts.Plots2d import Plots2d
from PlotScripts.Plots3d import Plots3d




def mainModelsInputs():
    device = selectDevice()
    dimension = int(input('Choose a dimension (2 or 3)!\n'))
    return loadConfigWithDim(dimension)


def mainModels(config: ConfigData, always_save = False, save_directory: str = None):
    ## DATA GENERATION
    # chooses if 2d or 3d
    if config.inputs==1:
        # 2D stuff
        data_cl = Data2d(config)
        plots_cl = Plots2d()
    elif config.inputs==2:
        # 3D stuff
        data_cl = Data3d(config)
        plots_cl = Plots3d()
    else:
        raise(ValueError('Given input size not implemented yet.'))


    ## MULTI NETS
    # multi net train and eval sochastic
    multi_net_data = multiNets(
        config, 
        data_cl, 
        plots_cl, 
        print_model=False, 
        print_train_log=False, 
        print_evals_b=False
    )


    ## SAVES
    # save figures and models 
    if always_save:
        save = 'y'
    else:
        plt.show(block=True)
        save = input('To save data, type in \'y\'.\n')
        
    if 'y' in save:
        if save_directory is not None:
            saves_cl = Saves(save_directory=save_directory)
        else:
            saves_cl = Saves(dimension=config.inputs+1)
        saves_cl.saveEverythingUnprun(config, *multi_net_data)
        plt.close()
        return saves_cl.get_save_directory()
    else:  
        plt.close()
    
    
    
    
if __name__ == "__main__":
    config = mainModelsInputs()
    make_latex_fonts()
    
    set_multi_thread()
    mainModels(config)
    del_multi_thread()