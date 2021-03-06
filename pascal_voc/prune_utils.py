def pre_prune_weights(self):
    # get weights in dict {name: torch.Tensor}
    state_dict = self.net.state_dict()
    threshold = 0.5
    # ================================================================ #
    # YOUR CODE HERE:
    #   1.find prunable variables i.e. kernel weight/bias
    #   2.prune parameters based on your threshold, calculated based on input argument percentage
    #   example pseudo code for step 2-3:
    #       for name, var in enumerate(state_dict):
    #           # construct pruning mask
    #           mask = var < threshold
    #           new_var = var[var < threshold]
    #           state_dict[name] = new_var
    # ================================================================ #
    for key, value in state_dict.items():
        if ("weight" in key or "bias" in key):
            mask = abs(value) > threshold
            new_value =  mask*value
            state_dict[key] = new_value

    self.net.load_state_dict(state_dict)
    return

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

def prune_weights_in_training(self):
# get weights in dict {name: torch.Tensor}
    state_dict = self.net.state_dict()
    threshold = 0.1
    # ================================================================ #
    # YOUR CODE HERE:
    #   you can reuse code for pre_prune_weights here
    #       -> make sure pruned weights not recovered
    #   or reselect threshold dynamically
    #       -> make sure pruned percentage same
    # ================================================================ #
    #c = 0
    for key, value in state_dict.items():
        if ("weight" in key or "bias" in key):  
            mask = abs(value) > threshold 
            new_value =  mask*value
            state_dict[key] = new_value
            #if c == 10:
            #    print('\n\n\n\nHERE', state_dict[key], '\n\n\n\n')
            #c +=1
    self.net.load_state_dict(state_dict)
    return

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

def prune_weights_in_training_threshold(self):
# get weights in dict {name: torch.Tensor}
    state_dict = self.net.state_dict()
    weight_sum = 0
    count = 0
    print('values', state_dict.values)
    for key, value in state_dict.items():
        if ("weight" in key or "bias" in key):
            weight_sum = weight_sum + 1
    threshold = 0.5

    # ================================================================ #
    # YOUR CODE HERE:
    #   you can reuse code for pre_prune_weights here
    #       -> make sure pruned weights not recovered
    #   or reselect threshold dynamically
    #       -> make sure pruned percentage same
    # ================================================================ #
    for key, value in state_dict.items():
        if ("weight" in key or "bias" in key):  
            mask = value > threshold
            new_value =  mask*value
            state_dict[key] = new_value
    return

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #