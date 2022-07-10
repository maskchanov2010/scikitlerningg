import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


train = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])
trains = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
suaisd = 2 * np.random.random((3, 1)) - 1

# print('wesa')
# print(suaisd)
for i in range(20000):
    ias = train
    owad = sigmoid(np.dot(ias, suaisd))
    err = trains-owad
    adfsf=np.dot( ias.T,err*(owad*(1-owad)) )
    suaisd+=adfsf
print('test')
print(owad)
# tr2 = array([
#     [1, 1, 1],
#     [0, 0, 0],
#     [1, 1, 1],
#     [0, 0, 0]
# ])
# tr1 = array([
#     [1, 1, 1, 1]
# ]).T
# random.seed(1)
# sy = 2*random.random(3,1)-2
# otr=1/(1+exp(-(tr1,sy)))
