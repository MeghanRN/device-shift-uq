from models.cpmobile import CPMobile
from models.dynacp import DynaCP
from models.gru_cnn import GRUCNN

def build_model(name: str, num_outputs: int):
    name = name.lower()
    if name in ("cpmobile","baseline"): 
        return CPMobile(num_outputs)
    if name in ("dynacp","interesting"): 
        return DynaCP(num_outputs)
    if name in ("gru-cnn","grucnn","cnn-gru","cnn_gru"): 
        return GRUCNN(num_outputs)
    raise ValueError(name)
