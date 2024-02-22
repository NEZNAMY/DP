friendly_to_code = {
    'Linear': 'linear',
    'Sigmoid': 'sigmoid',
    'Tanh': 'tanh',
    'ReLU (Rectified Linear Unit)': 'relu',
    'Leaky ReLU': 'leaky_relu',
    'Softmax': 'softmax',
    'Exponential Linear Unit (ELU)': 'elu',
    'Parametric ReLU (PReLU)': 'prelu',
    'Rectified Linear Unit with Parametric Noise (RReLU)': 'rrelu',
    'Scaled Exponential Linear Unit (SELU)': 'selu',
    'Swish': 'swish'
}

code_to_friendly = {v: k for k, v in friendly_to_code.items()}


def encode(friendlyName: str):
    return friendly_to_code.get(friendlyName, "UNKNOWN")


def decode(code: str):
    return code_to_friendly.get(code, "UNKNOWN")


def getFriendlyValues():
    return list(friendly_to_code.keys())
