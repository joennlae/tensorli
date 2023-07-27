from tensorli import Tensorli

def test_simple():
    x = Tensorli(3)
    y = Tensorli(4)
    z = x * y
    z.backward()
    assert x.grad.data[0] == 4
    assert y.grad.data[0] == 3
