def load_model_weights(model, weights, multi_gpus, train=True):
    """
        Load the model weights from the given checkpoint file
    """
    # If model was originally trained on a single GPU but needs to be loaded onto multiple ones,
    # it removes the "module" prefix from the weight keys
    if list(weights.keys())[0].find('module') == -1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True

    if (multi_gpus is False) or (train is False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights

    # load the model from the state_dict
    model.load_state_dict(state_dict)
    return model


# Class to work with if mixed precision is failing
class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


# Function to read CSS from file
def read_css_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()
