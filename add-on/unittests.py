import unittest
from .utils.websocket_utils import execute_operators_at_coordinates
from unittest.mock import patch

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

if __name__ == '__main__':
    unittest.main()