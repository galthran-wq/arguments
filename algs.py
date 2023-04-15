def get_subarray_index(A, B):
    """
    Find the best match of subarray A in array B.
    """
    n = len(A)
    m = len(B)
    # Two pointers to traverse the arrays
    i = 0
    j = 0
    best_match_length = 0
    match_length = 0
    best_start = None
    start = None

    # Traverse both arrays simultaneously
    while (i < n and j < m):

        # If element matches
        # increment both pointers
        if (A[i] == B[j]):
            if start is None:
                start = i
            
            match_length += 1
            i += 1;
            j += 1;

            # If array B is completely
            # traversed
            if (j == m):
                if match_length > best_match_length:
                    best_match_length = match_length
                    best_start = start
                return best_start;
        
        # If not,
        # increment i and reset j
        else:
            if match_length > best_match_length:
                best_match_length = match_length
                best_start = start
            start = None
            match_length = 0
            i = i - j + 1;
            j = 0;
        
    if match_length > best_match_length:
        best_match_length = match_length
        best_start = start
    return best_start;