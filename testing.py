import numpy as np
import numpy.ma as ma

mask = np.array([0,1,0,1])
values = np.array([1,2,3,4])

#res = np.multiply(arr, values)
res =  ma.masked_array(values,mask)
res = res[~res.mask]
print(res)