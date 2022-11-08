from descformats.base import PickleFile
import tempfile
import os
import pytest
from io import UnsupportedOperation


class PicklableTestClass:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z


def test_pickle_file():
    extra_provenance = {("top", "input"): "something"}
    obj = PicklableTestClass(1.0, "2", 3)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "something")
        with PickleFile("pk", path=path, mode="w", extra_provenance=extra_provenance) as p:
            p.write(obj)
            with pytest.raises(UnsupportedOperation):
                p.read()
                
        with PickleFile("pk", path=path, mode="r") as p:
            assert p.provenance[("top", "input")] == "something"
            obj2 = p.read()

            assert obj == obj2

            with pytest.raises(UnsupportedOperation):
                p.write(obj)

            
def test_mode_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "something")
        with pytest.raises(ValueError):
            p = PickleFile("pk", path=path, mode="q")
        with pytest.raises(ValueError):
            p = PickleFile("pk", path=path, mode="rw")
