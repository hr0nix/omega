import distrax


def entropy(logits):
    return distrax.Categorical(logits=logits).entropy()


def cross_entropy(labels, logits):
    return distrax.Categorical(probs=labels).cross_entropy(
        distrax.Categorical(logits=logits)
    )
