def calculate_precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

def calculate_recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)