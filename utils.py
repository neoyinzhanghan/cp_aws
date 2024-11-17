def smooth_function(x, beta=2.0):
    """
    Smoothly shifts the input value towards 1 using a power function.
    
    Parameters:
    - x (float): The input number between 0 and 1.
    - beta (float): The smoothing factor. Higher values make the shift sharper.
    
    Returns:
    - float: The transformed value closer to 1.
    """
    return 1 - (1 - x) ** beta
