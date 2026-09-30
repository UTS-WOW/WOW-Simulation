import socket
import threading

import numpy as np

from naval_rl.protocol import Connection


def test_roundtrip_through_a_real_socket():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    got = {}

    def serve():
        s, _ = server.accept()
        conn = Connection.__new__(Connection)
        conn.sock = s
        header, arrays = conn.recv()
        got["header"], got["arrays"] = header, {k: v.copy() for k, v in arrays.items()}
        conn.send({"type": "obs", "echo": header["type"]},
                  [("a", np.arange(6, dtype=np.float32).reshape(2, 3)), ("b", np.array([7, 8], dtype=np.int32))])
        s.close()

    t = threading.Thread(target=serve)
    t.start()
    c = Connection("127.0.0.1", port, connect_timeout=5)
    acts = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    c.send({"type": "step"}, [("actions", acts)])
    header, arrays = c.recv()
    t.join()
    server.close()

    assert got["header"]["type"] == "step"
    np.testing.assert_array_equal(got["arrays"]["actions"], acts)
    assert header["echo"] == "step"
    np.testing.assert_array_equal(arrays["a"], np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(arrays["b"], [7, 8])
