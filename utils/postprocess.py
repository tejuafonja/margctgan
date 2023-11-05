import sys
import numpy as np
import pandas as pd
from logger import get_logger

# Setup logger
LOGGER = get_logger('postprocess')

MAX_DECIMALS=sys.float_info.dig - 1

# Function from https://github.com/sdv-dev/RDT/blob/main/rdt/transformers/numerical.py
def learn_rounding_digits(data):
    # Check if data has any decimals
    name = data.name
    data = np.array(data)
    roundable_data = data[~(np.isinf(data) | pd.isna(data))]

    # Doesn't contain numbers
    if len(roundable_data) == 0:
        return None

    # Doesn't contain decimal digits
    if ((roundable_data % 1) == 0).all():
        return 0

    # Try to round to fewer digits
    if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
        for decimal in range(MAX_DECIMALS + 1):
            if (roundable_data == roundable_data.round(decimal)).all():
                return decimal

    # Can't round, not equal after MAX_DECIMALS digits of precision
    LOGGER.info("No rounding scheme detected for column '%s'. Data will not be rounded.", name)
    return None

if __name__ == "__main__":
    test_1 = [1.122222222222222222222, 1.2, 1.3, 1.234567]
    test_2 = [1.11, 1.2, 1.3, 1.234567]
    
    data = pd.DataFrame(test_2, columns=["data"])
    output = learn_rounding_digits(data.data)
    print(output)
    