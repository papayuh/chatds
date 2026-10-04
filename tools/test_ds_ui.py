"""Host guards for the real-window harness's ABI, buffers, and input mapping."""
import ctypes
import unittest
from unittest import mock

try:
    import ds_ui as ui
except OSError:  # CI without X11 runtime libraries cannot drive the GUI.
    ui = None


@unittest.skipIf(ui is None, 'X11 library unavailable')
class UIHarnessTest(unittest.TestCase):
    def conn(self):
        conn = ui.Conn.__new__(ui.Conn)
        conn.dpy = None
        conn.root = 1
        conn._layout = {}
        return conn

    def test_xlib_structures_include_fields_written_by_native_library(self):
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(ui.XWindowAttributes), 136)
            self.assertEqual(ui.XWindowAttributes.map_installed.offset, 88)
            self.assertEqual(ui.XWindowAttributes.all_event_masks.offset, 96)
            self.assertEqual(ui.XWindowAttributes.screen.offset, 128)
            self.assertEqual(ctypes.sizeof(ui.XInputEvent), 96)
        self.assertEqual(ctypes.sizeof(ui.XEvent), 24 * ctypes.sizeof(ctypes.c_long))

    def test_actual_leaf_client_not_decorator_or_selection_owner(self):
        c = self.conn()
        c.find_windows = lambda _: [(1, 'Qt Selection Owner'), (2, 'melonDS'), (3, 'melonDS')]
        c.geometry = lambda w: {1: (0,0,3,3,0), 2: (0,0,306,493,2), 3: (25,62,256,406,2)}[w]
        c._children = lambda w: [3] if w == 2 else []
        self.assertEqual(c.app_window('melonDS')[0], 3)

    def test_boot_guess_does_not_prevent_later_keyboard_calibration(self):
        c = self.conn()
        c._layout[3] = (256,406,0,11,1,'guess')
        c.grab = lambda _: (b'',256,406,1024)
        c._probe_layout = mock.Mock(return_value=(0,22,1))
        self.assertEqual(c.content_box(3)[:4], (0,22,1,'kbd'))
        c._probe_layout.assert_called_once()

    def test_touch_targets_client_coordinates_including_menu_offset(self):
        c = self.conn()
        c.content_box = lambda _: (0,22,1,'kbd',b'',256,406,1024)
        c.send_input = mock.Mock()
        with mock.patch.object(ui.time, 'sleep'):
            c.tap(3,236,152)
        self.assertEqual(c.send_input.call_args_list, [
            mock.call(3,4,1,236,366), mock.call(3,5,1,236,366,state=256)])

    def test_image_buffer_preserves_nuls_instead_of_reading_python_string_memory(self):
        class Image(ctypes.Structure):
            _fields_ = [('width',ctypes.c_int),('height',ctypes.c_int),
                ('xoffset',ctypes.c_int),('format',ctypes.c_int),('data',ctypes.c_void_p),
                ('byte_order',ctypes.c_int),('bitmap_unit',ctypes.c_int),
                ('bitmap_bit_order',ctypes.c_int),('bitmap_pad',ctypes.c_int),
                ('depth',ctypes.c_int),('bytes_per_line',ctypes.c_int),('bits_per_pixel',ctypes.c_int)]
        pixels = b'\x00\x7f\xff\x00\x03\x00\x05\xff'
        buf = ctypes.create_string_buffer(pixels)
        image = Image(width=2,height=1,data=ctypes.addressof(buf),
                      bytes_per_line=8,bits_per_pixel=32)
        def attributes(_dpy,_win,p):
            p._obj.width,p._obj.height = 2,1
            return 1
        c = self.conn()
        with mock.patch.object(ui,'XGetWindowAttributes',side_effect=attributes), \
             mock.patch.object(ui,'XGetImage',return_value=ctypes.addressof(image)), \
             mock.patch.object(ui,'XDestroyImage'):
            raw,w,h,stride = c.grab(3)
        self.assertEqual((raw,w,h,stride), (pixels,2,1,8))
        self.assertEqual(c._px(raw,stride,0,0), (255,127,0))


if __name__ == '__main__':
    unittest.main()
