import warnings




class DimensionException(Exception):
    def __init__(self, dimensions, message='Dimentions are not equal 2 or 3') -> None:
        self.dimensions = dimensions
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.dimensions} -> {self.message}'


class NotUniqueException():
    def __init__(self, duplicates, message='Duplicates found') -> None:
        self.duplicates = duplicates
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.duplicates} {self.message}'


class NotCorrectlyLoaded():
    def __init__(self, extra_mes, message='Data not correctly loaded') -> None:
        self.extra_mes = extra_mes
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.extra_mes} -> {self.message}'


class NotExistingChangingParameter():
    def __init__(self, layer_step, nodes_step, message='No steps to go there!') -> None:
        self.layer_step = layer_step
        self.nodes_step = nodes_step
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.layer_step} layer_step, {self.nodes_step} node_step -> {self.message}'
    



def set_error_to_warnings():
    warnings.simplefilter("default", UserWarning)
    warnings.simplefilter("always", UserWarning)
    warnings.filterwarnings("error", category=UserWarning)



def ignore_warning(message=''):
    warnings.filterwarnings("ignore", message=message)