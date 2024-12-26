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