import numpy as np


## Introduction ##
# Array = np.array([1,2,3,4])
# Array *= 2
# print(Array)


## Multidimenssional Arrays ##

# array1 = np.array([[1,2,3],
#                   [4,5,6],
#                   [7,8,9]])

# print(array1.ndim)


# array2 = np.array([[[1,2,3], [4,5,6], [7,8,9]],
#                    [[10,11,12], [13,14,15], [16,17,18]],
#                    [[19,20,21], [22,23,24], [25,26,27]]])

# print(array2.ndim)
# print(array2.shape)

# print(array2[0,0,2]) # its equal print(array2[0][0][2]) but the first type is faster this




## Slicing ##
#Array[strat:end:steps]

# array3 = np.array([[1,2,3,4],
#                    [5,6,7,8],
#                    [9,10,11,12],
#                    [13,14,15,16]])

# print(array3[0:2:1])
# print()
# print(array3[::2])
# print()
# print(array3[::-1])

# print(array3[2,1])
# print(array3[:,0])

# print(array3[:, 0:2]) 
# print(array3[:, 1::2]) 
# print(array3[0:2,0:2]) 




## Scalar Arithmetic ##

# array = np.array([1.5,2.32,3.097,4.8])

# print(array + 1)
# print(array - 2)
# print(array * 2)
# print(array / 3)
# print(array ** 4)

#Victorized math Functions
# print(np.sqrt(array))
# print(np.round(array))
# print(np.floor(array))
# print(np.ceil(array))
# print(np.pi)

#Element-Wise Arithmetic

# array1 = np.array([1,2,3])
# array2 = np.array([4,5,6])

# print(array1 + array2)
# print(array1 - array2)
# print(array1 * array2)
# print(array1 / array2)
# print(array1 ** array2)


#Comparsion
# Scores = np.array([92,35,68,98,78,88,22])

# print(Scores >50)

# Scores[Scores < 50] = 0
# print(Scores)








## BroadCasting ##

# array1 = np.array([1,2,3,4,5,6,7,8,9,10])
# array2 = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

# print(array1 * array2)






## Aggregate Functions ##

# array = np.array([[1,2,3,4,5],
#                   [6,7,8,9,10]])

# print(np.sum(array))
# print(np.mean(array))
# print(np.std(array)) #standard deviation
# print(np.var(array)) 
# print(np.argmin(array)) 
# print(np.argmax(array)) 
# print(np.sum(array, axis = 0)) #this sum all Columns
# print(np.sum(array, axis = 1)) #this sum all rows







## Filtering ##

# ages = np.array([[21,11,45,46,32,30,16,18,], [33,27,29,26,23,65,73,85]])

# teenagers = ages[ages < 18]
# adults = ages[(ages >= 18) & (ages < 65)]
# seniors = ages[ages >= 65]
# evens = ages[ages % 2 == 0]
# odds = ages[ages % 2 != 0]

# #another way

# adults = np.where(ages >= 18, ages, 0)

# print(adults)







## Random Numbers ##

# rng = np.random.default_rng() #rng = np.random.default_rng(seed = 1) # is if you want to save the same result

# print(rng.integers(1,10))

# print(rng.integers(low = 1, high = 10, size = (3,2)))

# print(np.random.uniform()) #this give a random float from 0 to 1 and you can control on range


# fruits = np.array(["Apple", "Banana", "Orange", "Mango"])
# fruits = rng.choice(fruits)
# fruits = rng.choice(fruits, size = (1,2))
# print(fruits)