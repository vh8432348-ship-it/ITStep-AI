import numpy as np
# Завдання 1
arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
#
# print(arr[3,1])
#
# print(arr[2,:])
#
# print(arr[:,0])
#
# print(arr[:2,:])
#
# arr[2:,:] = 100
# print(arr)
#
# arr[2,:] = arr[3,:]
# print(arr)

# Завдання 2
mask = arr % 2 == 0

print(arr[mask])

arr[mask] = 100

print(arr)


# Завдання 3
arr1 = np.array([128, 200, 10], dtype=np.uint8)
arr2 = np.array([250, 10, 34], dtype=np.uint8)

result = 0.2 * arr1 + 0.8 * arr2

result = np.round(result)

result = result.astype(np.uint8)

print(result.dtype)