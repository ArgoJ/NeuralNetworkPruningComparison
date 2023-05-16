import Pruning.PruningMethods.MagnitudePruning as PrunTypes



PRUN_TYPES = {
    # L1 unstructured
    'l1_unstr': PrunTypes.l1_unstructured_fixed_amount,
    'l1_unstr_val': PrunTypes.l1_unstructured_value_model,

    # L-inf structured
    'linf_str': PrunTypes.linf_structured_fixed_amount,
    'linf_str_val': PrunTypes.linf_structured_value,

    # L2 structured
    'l2_str': PrunTypes.l2_structured_fixed_amount,
    'l2_str_val': PrunTypes.l2_structured_value,

    # L1 structured
    'l1_str': PrunTypes.l1_structured_fixed_amount,
    'l1_str_val': PrunTypes.l1_structured_value,

    # global L1 unstructured
    'global_unstr': PrunTypes.global_unstructured_fixed_amount
}