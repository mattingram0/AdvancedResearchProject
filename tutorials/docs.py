import math
import sys

# Else Statements on Loops
for i in range(10):
    if i == 5:
        break
else:
    print("Not executed if we break out of the loop")


# Pass Statement
def placeHolder():
    pass


glob = 10


# Global Variables
def globalVars():
    global glob

# Python is Call by Value, where value is always an object reference

# All functions return a value - the default is None, which is often suppressed

# Default values are only ever evaluated once, and evaluated at the point of function
# definition in the defining scope

# When calling functions with default values, keyword argument pairs need to come after
# any non keyword argument pairs

# Tuple and Dictionary Parameters
def multParams(arg, *tuple, **dict):
    pass


multParams("arg", "arguments1", "arguments2", keyword1=1, keyword2=2)
# A dictionary parameter does not need to be specified. In this case, all additional
# parameters will end up in a tuple (otherArgs)
def arbParams(arg, *otherArgs):
    pass

# Unpacking argument lists - arguments already in a list/tuple of dictionary can be unpacked
# for a function call requiring separate positional (normal) parameters

args= [3, 6]
list(range(*args))  # range() expects (at least) 2 positional arguments, not a list

def unpackArgs(arg1, arg2="Arg 2", arg3="Arg 3"):
    pass

a = {"arg1": 1, "arg2": 2, "arg3": 3}
unpackArgs(**a)

# Lambda Expressions - syntactically restricted to a single expression. Like nested functions,
# they can reference variables from the containing scope. Lambda expressions can also be passed as args
def returnsAFunction(n):
    return lambda x: x + n

f = returnsAFunction(10)
print(f(5))  # Prints 15


# Function Annotations - optional metadata about parameter types and return types
def f(ham: str, eggs: str = 'eggs') -> str:
    print(f.__annotations__)
    return "Ham and Eggs"

# Using lists as stacks (LIFO) is efficeint, but using lists as queues (FIFO)
# is not

# List comprehension is preferable because it has no side effects (no
# variables left over)
# The syntax is: [expr for ... for ... if ...]
# Variables cannot be used in later for expressions if they have not been
# defined in previous for expressions

# Nested List Comprehensions
matrix = [[0 for i in range(5)] for j in range(5)]
[[row[i] for row in matrix] for i in range(5)]  # Transpose a matrix

# del removes a value from a list given its index, but does not return a
# value (like pop) does

# tuples are immutable, lists are mutable

singleton = "hello",  # A one-element tuple - note the ugly comma
t = 12345, 54321, "hello"  # Tuple packing - no parenthesis needed
x, y, z = t  # Sequence unpacking

# When we have a GUI, or lots of state, use OOP. If we are simply processing
# input data and generating output data, use declarative (functional)
# programming

# Tuples can be keys in dictionaries providing they contain only other
# tuples or strings or numbers

# Dictionary Comprehension
{x: x**2 for x in (2, 4, 6)}

# Dictionary creation using keyword arguments
a = dict(a=1234, b=5678, c=9)

# Looping through dictionaries:
for k, v in a.items():
    print(k, v)

# Looping through sequences:
l = [1, 2, 3, 4, 5]
for i, v in l:
    print(i, v)

# sorted() returns a new sorted list, whilst leaving the original source
# unaltered. sort() sorts the original list in place

# Sequences/string are compared lexicographically - that is, pairwise

# The module name is the name of the file without the .py. Accessible using
# the __name__ global variable

# Executable code in a module (a file used as in import) is executed only
# the first time the module name is encountered in an import statement.

# Packages are groups of modules, usually organised in a directory
# structure. __init__.py is needed in each folder in the directory structure
# to ensure that python knows to treat this directory as a package -
# __init__.py can be empty, or it can execute initialization code. For example

# sound/
#   __init__.py
#   formats/
#       __init__.py
#          wavread.py

# import sound.formats.wavread

# The different kinds of imports:
# import sound.formats.wavread <- functions must be referenced using the
# full name
# from sound.formats import wavread <- functions can be used without package
# prefix

# Formatted string literals
year = 2016
event = "Referendum"
print(f'Results of the {year} {event}')
print(f'The value of pi is approx {math.pi:.3f}.')

#str.format()
yesVotes = 80
percentage = 80/111
print('{:-9} Yes votes - {:2.2%}'.format(yesVotes, percentage))
print('The story of {0}, {1}, and {other}.'.format('Bill', 'Manfred',
                                                   other='Georg'))
# Always open files using the binary mode unless the file is a text file
# Use the with keyword paradigm. If you don't use with, ensure you call
# f.close()
# f.read() reads the entire contents of a file - coder problem to ensure it's
# not enourmous. f.read(size) reads at most size text characters/bytes

# Looping over lines in a file (memory efficient and fast):
with open('file') as f:
    for line in f:
        print(line, end='')

f.tell()  # Return the number of bytes from the start the file object
# currently is at
f.seek(5, 0)  # Go to the 6th byte from the start (0) of the file

# JSON can be used for serialization/deserialization and inter-application
# exchange. Pickle is python only, and can be used for serialization of
# arbitrarily complex Python objects.

# In syntax errors, the error is detected at the token preceding the token
# that is pointed to by the carat symbol in the syntax error.

# The try clause is executed. If no exception occurs, the except clause is
# skipped. If one occurs, then if there is a matching except, this is
# executed, and execution returns to the next line after the try/except
# block. If one occurs and is not handled in an except clause, execution
# stops and we have an unhandled exception

while True:
    try:
        x = int(input("Please enter a number: "))
        break
    except (RuntimeError, ValueError):  # Handle multiple exceptions
        print("That wasn't a valid number")
    except:
        print("Unexpected error: ", sys.exec_info()[0])  # Catch all other exc
        raise  # Re-raises the original exception
    else:
        print("Executed only if the try clause does not raise an exception")
    finally:
        print("This is always executed")

try:
    raise Exception("arg1", "arg2")
except Exception as exc:
    print(exc.args)
    print(exc)  # __str__ is defined for all exceptions, so can be printed
    # directly
    x, y = exc.args  # Exception arguments can be accessed

# When defining our own exceptions, typically have one class called
# class Error(Exception):, and then have class SpecificError(Error) that
# extends this base error class with specific functionality

# Normally, class members are public (except private variables), and all
# member functions are virtual.

# Objects have individualiyt, and multiple names can be bound to the same
# object - known as aliasing in other languages. Passing an object is cheap
# since only a pointer is passed, and if the function modifies the object,
# the casser will see the change. PASS BY REFERENCE

# A namespace is a mapping from names to objects. There is no relation to
# names in different namespaces. Two modules may both define a function
# maximize without confusion - users must prefix it with the module name
# Each module has its own global namespace. The namespace containing
# built-in names is created when the python interpreter starts, and is never
# deleted. A local namespace for a function is created when the function is
# called, and deleted when the function raises an exception or returns
# The global statement can be used to indicate that particular variables live
# in the global scope and should be rebound there; the nonlocal statement
# indicates that particular variables live in an enclosing scope and should
# be rebound there.

def spam_scope():

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_global()
    print(spam)  #prints "test spam" - the global spam variable effects only
    # the module's global namespace, and not any of the other local function
    # namespaces

# A new namespace is created and used as the local scope inside class
# definitions. When a class definition is left normally, a class object is
# created

class MyClass:
    """A simple example class"""
    i = 123  # Class variable - shared by all instances
    def __init__(self, val):
        self. i = val  # Instance variable unique to each instance

    def f(self):  # Function object
        return 'hello world'

x = MyClass()  # Object instantiation - automatically invokes __init__()

# There are two kinds of attribute references:
# - Data attributes in python correspond to 'instance variables' in other
#   languages - they DON'T NEED TO BE DECLARED, and like local variables,
#   spring into existence when they are first assigned to
# - Method attributes are functions that 'belong to' an object
# x.f is not the same as MyClass.f - is is a method object, not a function
# object. Method objects can be stored and called at a later time

xf = x.f  # The function is not called here
xf()  # It is called here, and the instance object is passed as the first
# argument to the function

class GoodDog:
    def __init__(self, name):
        self.name = name
        self.tricks = []

    def add_trick(self, trick):
        self.tricks.append(trick)


class BadDog:
    def __init__(self, name, tricks=[]):
        self.name = name
        self.tricks = tricks

    def add_trick(self, trick):
        self.tricks.append(trick)

def run():
    g = GoodDog('Gary')
    g.add_trick("Roll over")
    print(g.tricks)
    g.__init__('Gary')
    g.add_trick("Play dead")
    print(g.tricks)

    b = BadDog('Billy')
    b.add_trick("Roll over")
    print(b.tricks)
    b.__init__('Billy')  # DEFAULT ARGUMENT IS NOT RESET
    b.add_trick("Play dead")
    print(b.tricks)

# Any function object that is a class attribute defines a method for instances
# of that class. It is not necessary that the function definition is textually
# enclosed in the class definition.

# Inheritance
class ReallyGoodDog(GoodDog):  # This is a derived class from the base class
    def callBase(self):
        GoodDog.add_trick(self, "Sit")

# Multiple inheritance
class OddDog(GoodDog, BadDog):
    pass

# In multiple inheritence, attributes are searched for first in GoodDog (
# recursively), then BadDog (recursively). Techinically this is not true - the
# searching method is dynamic to supprt cooperative calls to super

#"Structs"
class Employee:
    pass


john = Employee()
john.name = "John Doe"  # Data attributes do not need to be declared
john.dept = "Computer Lab"

# Iterators
# Behind the scenes, for calls iter() on the container object that it is
# iterating over. The function returns an iterator object, that defines the
# method __next__(), which accesses elements in the container one at a time
# until there are no more elements, when __next__() raises a StopIteration
# exception which tells the for loop to terminate. next() calls the
# __next__() function.


class Reverse:  # Iterator used to return a collection
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):  # Must return an object with the __next__ function
        return self  # This class does have a __next__ function so is a
        # valid iterator

    def __next__(self):
        if self.index == 0:
            raise StopIteration  # Raise a StopIteration when we have
            # reached the beginning of the iterator
        self.index = self.index - 1

        return self.data[self.index]

# Generators
# Simple, powerful tool for creating iterators. Written like regular
# functions but use YIELD when they want to return data. Each time next() is
# called on a generator, the generator resumes where it left off (remember
# all the data values and which satement was last executed)
# Anything that can be done with generators can also be done with class
# based iterators - what make generators powerful is that __iter__() and
# __next__() are created automatically
# Local variables and execution state are automatically saved between calls
# - easier to write than using instanvce variables like self.index

def reverse(data):
    for i in range(len(data) - 1, -1, -1):
        yield data[i]

# Generator expressions
# Simple generators can be coded succinctly as expression using a syntax
# similar to list comprehension, but using parentheses instead of square
# brackets.

sum((i*i for i in range(10))) #(i*i for i in range(10)) is a generator,
# which creates an iterable object that has a next method. This is required
# because sum() expects an iterable object as its argument


