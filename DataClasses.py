from matplotlib.figure import Figure
from graphviz import Digraph
from pyx.canvas import canvas


from NeuralNetwork import Model



class MetricFigures(list[Figure]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class ModelFigures(list[list[Figure]]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class Models(list[list[tuple[list[int], Model]]]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class ModelsNotRemovedNodes(list[list[tuple[list[int], Model]]]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class Graphs(list[list[Digraph]]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class Draws(list[list[canvas]]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class IterPrunFigures(list[Figure]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class PerformanceFigures(list[Figure]):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class OriginalFigure(Figure):
    pass