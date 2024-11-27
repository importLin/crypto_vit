def accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = preds.size(0)
    acc = correct / total
    return acc