import numpy as np
# Завдання 1
arr = np.arange(1, 11)

print("Масив:", arr)
print("Розмір:", arr.shape)
print("Тип:", arr.dtype)

arr = arr.reshape(5, 2)

print(arr)
print("Розмір:", arr.shape)
print("Тип:", arr.dtype)
# Завдання 2

arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print(arr[1, 2])
print(arr[:,-1])
print(arr[:, 2:])
print(arr[:2, 2:])
arr[:2, 2:] = -1
print(arr)
arr[:, 0] = arr[:, 1]
print(arr)

# Завдання 3

mask = arr > 6

print("Маска:")
print(mask)


print("Кількість:", np.sum(mask))


print(arr[mask])

arr1 = arr.copy()
arr1[mask] += 10

print(arr1)

arr2 = arr.copy()
arr2[~mask] *= -1

print(arr2)

replace_arr = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

arr3 = arr.copy()
arr3[mask] = replace_arr[mask]

print(arr3)

# Завдання 6



arr = np.array([10, 4, 25, 40, 200], dtype=np.uint8)

print(arr)
print(arr.dtype)

arr2 = np.clip(arr * 2, 0, 255).astype(np.uint8)

print(arr2)
print(arr2.dtype)

arr3 = np.clip(arr * 1.5, 0, 255).astype(np.uint8)

print(arr3)
print(arr3.dtype)