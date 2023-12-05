import numpy as np

def get_average_intensity(indices,point_colors):
    #If indices is a NumPy array with more than one dimension, flatten it
    if isinstance(indices, np.ndarray) and indices.ndim > 1:
        indices = indices.flatten()

    #If indices is a scalar, convert it to a list with a single element
    if np.isscalar(indices):
        indices = [indices]

    total_intensity = 0.0
    point_amount = len(indices)
    for index in indices:
        
        intensity = np.average(point_colors[index])
        total_intensity += intensity

    return total_intensity / point_amount

