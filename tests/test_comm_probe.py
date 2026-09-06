import socket
import unittest
from unittest.mock import MagicMock, patch

from comm.comm_probe import measure_comm_matrix


class ProbeFailureTests(unittest.TestCase):
    def run_failed_pair(self, rank, bind_failure=False, remote_unavailable=False):
        store = MagicMock()
        values = {}
        store.set.side_effect = values.__setitem__
        store.get.side_effect = lambda key: values.get(
            key, (b'0' if remote_unavailable else b'1')
            if key.startswith('probe-ready-') else b'127.0.0.1')
        listener = MagicMock()
        if bind_failure:
            listener.bind.side_effect = OSError('address already in use')
        listener.accept.side_effect = socket.timeout('no peer connected')

        def gather(output, local_result):
            output[:] = [local_result, {'rank': 1 - rank, 'measurements': []}]

        with patch('comm.comm_probe.dist') as dist, \
                patch('comm.comm_probe._guess_ipv4', return_value='127.0.0.1'), \
                patch('comm.comm_probe.socket.socket', return_value=listener), \
                patch('comm.comm_probe.socket.create_connection',
                      side_effect=socket.timeout('connection timed out')) as connect:
            dist.distributed_c10d._get_default_store.return_value = store
            dist.all_gather_object.side_effect = gather
            latency, bandwidth = measure_comm_matrix(None, rank, 2, timeout=.1)

        self.assertEqual(latency, [[0.0, None], [None, 0.0]])
        self.assertEqual(bandwidth, latency)
        listener.close.assert_called_once()
        dist.all_gather_object.assert_called_once()
        if remote_unavailable:
            connect.assert_not_called()
        else:
            connect.assert_called_once()

    def test_connection_timeout_reaches_aggregation(self):
        self.run_failed_pair(rank=0)

    def test_listener_bind_failure_reaches_aggregation(self):
        self.run_failed_pair(rank=1, bind_failure=True)

    def test_unavailable_listener_skips_connection(self):
        self.run_failed_pair(rank=0, remote_unavailable=True)


if __name__ == '__main__':
    unittest.main()
