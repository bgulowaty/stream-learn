# from typeguard import typechecked
# from enforce import runtime_validation as typechecked
from attr import attrs, attrib
from pytypes import typechecked


@typechecked
@attrs
class A:
    a: int = attrib()
    b: str = attrib()
    c: list = attrib(default=[])


# @attrs
# class B:
#     c: int = attrib()
#
#
# @typechecked
# @attrs
# class C:
#     a: A = attrib()


print(A(123, b='23'))

# print(C('xd'))
