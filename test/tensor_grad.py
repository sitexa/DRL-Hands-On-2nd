from sympy import symbols, diff

# 定义变量和函数
x, y = symbols('x y')
f = x ** 2 + 2 * x * y + y ** 2

# 计算对 x 的偏导数
df_dx = diff(f, x)
print("Partial derivative of f with respect to x:", df_dx)

# 计算对 y 的偏导数
df_dy = diff(f, y)
print("Partial derivative of f with respect to y:", df_dy)
