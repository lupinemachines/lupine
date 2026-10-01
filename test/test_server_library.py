import ctypes
import socket
import subprocess
import sys

if len(sys.argv) == 3:
    try:
        library = ctypes.CDLL(sys.argv[1])
    except OSError as error:
        if "libcuda" in str(error) or "undefined symbol: cu" in str(error):
            sys.exit(77)  # CPU builders may provide incomplete CUDA stubs.
        raise
    serve = library.lupine_server_serve_connection_v1
    serve.argtypes = [ctypes.c_int]
    serve.restype = ctypes.c_int
    sys.exit(serve(int(sys.argv[2])))

# Exercise the real entrypoint in an otherwise ordinary embedding process.
for socket_type in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
    client, worker = socket.socketpair(type=socket_type)
    with client, worker:
        with subprocess.Popen([sys.executable, __file__, sys.argv[1], str(worker.fileno())],
                              pass_fds=(worker.fileno(),)) as process:
            worker.close()
            if socket_type == socket.SOCK_STREAM:
                client.settimeout(5)
                client.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                try:
                    response = client.recv(4096)
                except ConnectionResetError:
                    response = b""
                if not response and process.wait(timeout=5) == 77:
                    sys.exit(77)
                assert response.startswith(b"HTTP/1.1 200"), response
            result = process.wait(timeout=5)
            assert result == (0 if socket_type == socket.SOCK_STREAM else 1), result
