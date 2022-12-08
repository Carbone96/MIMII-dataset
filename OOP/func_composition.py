import functools
from typing import Callable

ComposableFunction = Callable[[float], float]

def compose( *functions: ComposableFunction) -> ComposableFunction:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)

def addThree(x : float) -> float:
    return x + 3

def multiplyByTwo(x : float) -> float:
    return x * 2

def main() -> None:
    x=12
    myfun = compose(multiplyByTwo,addThree,multiplyByTwo,addThree)
    result = myfun(x)
    print(f"my result is {result}")

if __name__ == "__main__":
    main()