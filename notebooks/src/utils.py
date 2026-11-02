import numpy as np
import copy
import jax
import jax.numpy as jnp

def stratified_split(key,
                     dataset,
                     num_samples_per_class):
    key_choices,key_shuffle = jax.random.split(key,2)
    key_choices = jax.random.split(key_choices,len(dataset.classes))
    dataset_size = len(dataset)
    indices = jnp.arange(dataset_size)
    targets = jnp.array(dataset.targets)
    stratified_sample_idx = []

    for label,label_idx in dataset.class_to_idx.items():
        label_idxs = indices[targets==label_idx]
        rnd_label_idxs = jax.random.choice(
            key_choices[label_idx],
            label_idxs,
            (num_samples_per_class,),
            replace=False)
        stratified_sample_idx.append(rnd_label_idxs)

    # stratified_sample_idx = jax.random.shuffle(
    #     key_shuffle,
    #     jnp.array(stratified_sample_idx).ravel()) 
    stratified_sample_idx = jax.random.permutation(
        key_shuffle, 
        jnp.array(stratified_sample_idx).ravel(), independent=True)
    other_sample_idx = jnp.delete(
        indices,
        stratified_sample_idx)

    dataset_copy1 = copy.deepcopy(dataset)
    dataset_copy2 = copy.deepcopy(dataset)
    dataset_copy1.data = dataset.data[stratified_sample_idx]
    dataset_copy1.targets = targets[stratified_sample_idx]
    dataset_copy2.data = dataset.data[other_sample_idx]
    dataset_copy2.targets = targets[other_sample_idx]
    return dataset_copy1,dataset_copy2