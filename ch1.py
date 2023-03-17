print(5 + 6)
print(2.5 / 3)
print(5 == 6)
print(5 < 6)
print((5 < 6) and not (5 == 6))
print(False or True)
print(True or False)
print('hello')
print(len('hello'))
print('hello' + ' ' + 'world')
print('h' in 'hello')
x = 5
y = 7
print(x + y)
book_title = 'Hands-On Data Analysis with Pandas'
print(type(x))
print(type(book_title))
print(book_title)
var = ['hello', ' ', 'world']
my_list = ['hello', 3.8, True, 'Python']
print(type(my_list))
print(len(my_list))
print('world' in my_list)
print(my_list[1])
print(my_list[-1])
print(my_list[1:3])
print(my_list[::2])
print(my_list[::-1])
print('|'.join(['x', 'y', 'z']))
my_tuple = ('a', 5)
print(type(my_tuple))
print(my_tuple[0])
shopping_list = {
    'veggies': ['spinach', 'kale', 'beets'],
    'fruits': 'bananas',
    'meat': 0
}
print(type(shopping_list))
print(shopping_list['veggies'])
print(shopping_list.keys())
print(shopping_list.values())
print(shopping_list.items())
my_set = {1, 1, 2, 'a'}
print(type(my_set))
print(len(my_set))
print(my_set)
print(2 in my_set)


def add(x, y):
    return x + y


type(add)
add(1, 2)
result = add(1, 2)
result
print_result=print('hello world')
type(print_result)
print_result is None

def make_positive(x):
    """Returns a positive x"""
    if x<0:
        x*=-1
    return x

make_positive(-1)
make_positive(2)

def add_or_subtract(operation, x, y):
    if operation== 'add':
        return x+y
    else:
        return x-y

add_or_subtract('add', 1,2)
add_or_subtract('subtract',1,2)

def calculate(operation,x,y):
    if operation =='add':
        return x+y
    elif operation =='subtract':
        return x-y
    elif operation== 'multiply':
        return x*y
    elif operation =='division':
        return x/y
    else:
        print("This case hasn't been handled")

calculate('multiply',3,4)
calculate('power',3,4)

done=False
value=2
while not done:
    print('Still going...',value)
    value*=2
    if value>10:
        done=True

value=2
while value<10:
    print('Still going...', value)
    value*=2

for i in range(5):
    print(i)

for element in my_list:
    print(element)

for key, value in shopping_list.items():
    print('For',key,'we need to buy',value)

import math
print(math.pi)

from math import pi
print(pi)


class Calculator:
    """This is the class docstring."""

    def __init__(self):
        """This is a method and it is called when we create an object of type `Calculator`."""
        self.on = False

    def turn_on(self):
        """This method turns on the calculator."""
        self.on = True

    def add(self, x, y):
        """Perform addition if calculator is on"""
        if self.on:
            return x + y
        else:
            print('the calculator is not on')

my_calculator = Calculator()

my_calculator.add(1, 2)

my_calculator.turn_on()
my_calculator.add(1,2)
my_calculator.on
my_calculator.on=False
my_calculator.add(1,2)

import random

random.seed(0)
salaries = [round(random.random()*1000000, -3) for _ in range(100)]

##mean
import statistics as s
s.mean(salaries)

##median
s.median(salaries)

##mode
s.mode(salaries)

##sample variance
s.variance(salaries)

##sample standard deviation
s.stdev(salaries)


#range
min=min(salaries)
max=max(salaries)
range=max-min
range

#coeffiecient of variation
#sd/mean
coefv=s.stdev(salaries)/s.mean(salaries)
coefv

#interquartile range
import numpy as np
q1=np.percentile(salaries,25,interpolation='midpoint')
q3=np.percentile(salaries,75,interpolation='midpoint')
iqr=(q3-q1)
iqr

#quartile coefficient of dispersion
#iqr/(q1+q3)
qcd=iqr/(q1+q3)
qcd

#min-max scaling
min_max_scaled=[(x-min)/range for x in salaries]
min_max_scaled[:5]

#standardising
means=s.mean(salaries)
sds=s.stdev(salaries)
standard=[(x-means)/sds for x in salaries]
standard[:5]

#covariance
cov=np.cov(min_max_scaled, standard)

#pearson correlation coefficient
cov/(s.stdev(min_max_scaled)*s.stdev(standard))

