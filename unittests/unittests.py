import unittest
from ..scripts.full_script_websocket import execute_operators_at_coordinates, WebSocketServerManager,update_connect_socket, websocket_port, context,run_asyncio_coroutine, handle
from unittest.mock import patch
from unittest.mock import patch, MagicMock
import asyncio

class TestExecuteOperatorsAtCoordinates(unittest.TestCase):

    @patch('your_module.bpy.ops.custom.mark_fixed_triangle')
    def test_triangle_marker(self, mock_mark_fixed_triangle):
        execute_operators_at_coordinates(1, 2, 3, "triangle_marker")
        mock_mark_fixed_triangle.assert_called_with(external_x=1, external_y=2, external_z=3)

    @patch('your_module.bpy.ops.custom.mark_complex')
    def test_complex_marker(self, mock_mark_complex):
        execute_operators_at_coordinates(4, 5, 6, "complex_marker")
        mock_mark_complex.assert_called_with(external_x=4, external_y=5, external_z=6)

    @patch('your_module.print')
    def test_no_marker(self, mock_print):
        execute_operators_at_coordinates(0, 0, 0, "None")
        mock_print.assert_called_with("No marker selected")

    @patch('your_module.print')
    def test_unknown_marker(self, mock_print):
        execute_operators_at_coordinates(0, 0, 0, "unknown_marker")
        mock_print.assert_called_with("Unknown marker type")

class TestWebSocketServerManager(unittest.TestCase):

    @patch('your_module.threading.Thread')
    @patch('your_module.asyncio.new_event_loop')
    def test_start_server(self, mock_new_event_loop, mock_thread):
        manager = WebSocketServerManager(8080)
        manager.start_server()
        mock_new_event_loop.assert_called()
        mock_thread.assert_called()

    @patch('your_module.threading.Thread')
    def test_stop_server(self, mock_thread):
        # Mocking the necessary components
        manager = WebSocketServerManager(8080)
        manager.loop = MagicMock()
        manager.loop.is_running.return_value = True
        manager.server = MagicMock()
        manager.thread = MagicMock()
        manager.thread.is_alive.return_value = True

        manager.stop_server()

        # Check if the server is stopped and the loop is terminated
        manager.loop.call_soon_threadsafe.assert_called_with(manager.server.close)
        manager.loop.call_soon_threadsafe.assert_called_with(manager.loop.stop)
        manager.thread.join.assert_called_once()

    @patch('your_module.WebSocketServerManager.stop_server')
    @patch('your_module.WebSocketServerManager.start_server')
    def test_restart_server(self, mock_start_server, mock_stop_server):
        manager = WebSocketServerManager(8080)
        manager.restart_server()

        # Ensure that stop_server and start_server are called in sequence
        mock_stop_server.assert_called_once()
        mock_start_server.assert_called_once()

@patch('your_module.WebSocketServerManager')
def test_update_connect_socket(self, mock_server_manager):
    # Assuming 'self' and 'context' are provided or mocked appropriately
    self.connect_socket = True
    update_connect_socket(self, context)
    mock_server_manager.assert_called_with(port=websocket_port)
    mock_server_manager.restart_server.assert_called_once()

    self.connect_socket = False
    update_connect_socket(self, context)
    mock_server_manager.stop_server.assert_called_once()

@patch('your_module.asyncio')
def test_run_asyncio_coroutine(self, mock_asyncio):
    test_coroutine = MagicMock()
    run_asyncio_coroutine(test_coroutine)
    mock_asyncio.new_event_loop.assert_called()
    mock_asyncio.set_event_loop.assert_called()
    test_coroutine.run_until_complete.assert_called_with(test_coroutine)
    
@patch('your_module.json.loads')
@patch('your_module.execute_operators_at_coordinates')
async def test_handle(self, mock_execute, mock_json_loads):
    mock_websocket = MagicMock()
    mock_websocket.recv = MagicMock(return_value=asyncio.Future())
    mock_websocket.recv.return_value.set_result('{"message": {"x": 1, "y": 2, "z": 3, "type": "triangle_marker"}}')

    mock_json_loads.return_value = {"message": {"x": 1, "y": 2, "z": 3, "type": "triangle_marker"}}

    await handle(mock_websocket, "/test")
    mock_execute.assert_called_with(1, 2, 3, "triangle_marker")


if __name__ == '__main__':
    unittest.main()