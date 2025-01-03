import numpy as np

def array_to_markdown(arr: np.ndarray) -> str:
    """
    Convert a 1D or 2D NumPy array into a Markdown table.
    
    Parameters
    ----------
    arr : np.ndarray
        The NumPy array to convert. Must be 1D or 2D.
    
    Returns
    -------
    str
        A string containing the Markdown table.
    """
    
    if arr.ndim == 1:
        # 1D array -> "Index | Value" table
        out = "| Index | Value |\n| --- | --- |\n"
        for i, val in enumerate(arr):
            out += f"| {i} | {val} |\n"
        return out
    
    elif arr.ndim == 2:
        # 2D array -> "Col 0 | Col 1 | ..." plus rows
        rows, cols = arr.shape
        
        # Header row
        header = [f"Col {j}" for j in range(cols)]
        out = "| " + " | ".join(header) + " |\n"
        
        # Separator row
        out += "| " + " | ".join(["---"] * cols) + " |\n"
        
        # Data rows
        for i in range(rows):
            row_str = " | ".join(str(val) for val in arr[i])
            out += f"| {row_str} |\n"
            
        return out
    else:
        raise ValueError("Only 1D or 2D arrays are supported by this function.")

def lexicographic_comparison(a : list | np.ndarray, b: list | np.ndarray) -> int:
    def first_element_is_list(iterable):
        condition = lambda x: type(x) == list or type(x) == np.ndarray
        return next(
            (i for i in iterable if condition(i)), None
        )
    if first_element_is_list(a) or first_element_is_list(b):
        raise Exception('Error, arrays must be 1-dimensional only')
    if len(a) != len(b):
        raise Exception('Error, arrays must have the same length')
    for a_, b_ in zip(a, b):
        if a_ > b_:
            return 1
        elif a_ < b_:
            return -1
    return 0
