Question  # 1

# Change name to Rohan without using replace().
name = 'Mohan'
name = 'Rohan'  # just replace the asssignmets

Question  # 2

# Swap the 2 variables below without using a third variable.

a = 7

b = 10

a, b = b, a

Question  # 3


# Can you point out the mistake

def test(a=10, b):
    pass


# default parameters are used in last


Question  # 4


# write code for factorial as a recursive function


def fact(n):
    i = 1


for i in range(n):
    return fact(i) * fact(i - 1)

Question  # 5

# What will be the value of b?

a = []

b = a * 4
b = []

Question  # 6

# What will be the value of b?

a = [[1], [2]]

b = a * 4

b = [[1], [2], [1], [2], [1], [2], [1], [2]]

Question  # 7

# write the output

list = ['a', 'b', 'c', 'd', 'e']

print(list[10:])

# idex out of range error


Question  # 8

cars = [{'model': 'Tesla', 'Reviews': 300}, {'model': 'Toyota', 'Reviews': 100}]

car_map = map(lambda x: x[‘model
'], cars)

print(list(car_map))
[{'model': 'Tesla'}, {'model': 'Toyota'}]

Question  # 9

list_a = [1, 2, 3]
list_b = [10, 20, 30]

temp = map(lambda x, y: x + y, list_a, list_b)
print(list(temp))
[11, 22, 33]

Question  # 10

a = a = [16, 20, 3, 4, 5, 6]
f = filter(lambda x: x % 2 == 0, a)
print(list(f))

[16, 20, 4, 6]