"""Read local Docker state without waiting on CLI log/config machinery."""
import http.client,json,socket
from urllib.parse import quote
class LocalDocker(http.client.HTTPConnection):
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/run/docker.sock')
def inspect_container(name):
    connection=LocalDocker('localhost',timeout=10)
    try:
        connection.request('GET','/containers/'+quote(name,safe='')+'/json')
        response=connection.getresponse()
        if response.status==404:return None
        if response.status!=200:raise RuntimeError('Docker inspect status '+str(response.status))
        return json.load(response)
    finally:connection.close()
