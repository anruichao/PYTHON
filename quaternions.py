'''

This is the quaterions.py sample code to get you started.

'''
import numpy as np


class quat:
    def __init__(self, *args):
        if len(args) == 4:
            self.a = args[0]
            self.b = args[1]
            self.c = args[2]
            self.d = args[3]
        elif len(args) == 1:
            self.a = args[0]
            self.b = 0
            self.c = 0
            self.d = 0
        else:
            raise ValueError('Wrong number of inputs to constructor.')

    def __str__(self):
        return str(self.a) + '+' + str(self.b) + 'i+' + str(self.c) + 'j+' + str(self.d) + 'k'

    def __add__(self, other):
        if type(other) == quat:
            return quat(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        else:
            return self + quat(other)

    def __mul__(self, other):
        if type(other) == quat:
            x = self.a
            y = self.b
            z = self.c
            w = self.d
            qw = other.d
            qx = other.a
            qy = other.b
            qz = other.c
            # rw = w * qw - x * qx - y * qy - z * qz
            # rx = w * qx + x * qw + y * qz - z * qy
            # ry = w * qy + y * qw + z * qx - x * qz
            # rz = w * qz + z * qw + x * qy - y * qx
            rw = w * qw + x * qw + y * qw + z * qw
            rx = x * qx + y * qx + z * qx + w * qx
            ry = w * qy + y * qy + z * qy + x * qy
            rz = w * qz + z * qz + y * qz + x * qz
            return quat(rx, ry, rz, rw)

        else:
            return self * quat(other)

    def __rmul__(self, other):
        x = self.a
        y = self.b
        z = self.c
        w = self.d
        if type(other) == int:
            rw = w * other
            rx = x * other
            ry = y * other
            rz = z * other
            return quat(rx, ry, rz, rw)

    def __sub__(self, other):
        if type(other) == quat:
            return quat(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)
        else:
            return self - quat(other)

    def norm(self):
        x = self.a
        y = self.b
        z = self.c
        w = self.d
        length = x * x + y * y + z * z + w * w
        if length != 1 and length != 0:
            length = 1.0 / np.sqrt(length)
            return quat(length * x, length * y, length * z, length * w)
        return quat(x, y, z, w)

    def inv(self):
        x = self.a
        y = self.b
        z = self.c
        w = self.d
        # length = x * x + y * y + z * z + w * w
        # if length != 1 and length != 0:
        #     length = 1.0 / np.sqrt(length)
        #     # print(length)
        #     return quat(-length * x, -length * y, -length * z, length * w)
        # return quat(-x, -y, -z, w)
        if (x != 0): x = float('%0.2f'%(1 / x))
        if (y != 0): y = float('%0.2f'%(1 / y))
        if (z != 0): z = float('%0.2f'%(1 / z))
        if (w != 0): w = float('%0.2f'%(1 / w))
        return quat(x, y, z, w)



x = quat(5)
y = quat(0, 4, 0, 0)
z = quat(1, 1, 1, 1)
print(x)
print(x + y)
print(x - y)
print(2 * x)
print(x.inv())
print(x * y)
print(x * z)



''' Sample output:
5+0i+0j+0k
5+4i+0j+0k
5+-4i+0j+0k
10+0i+0j+0k
0.2+0.0i+0.0j+0.0k
0+20i+0j+0k
5+5i+5j+5k
'''
