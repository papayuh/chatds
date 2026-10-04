#!/usr/bin/env python3
"""Real-melonDS UI automation with targeted X11 events (ctypes, no extra deps).

Events traverse the actual Qt client -> melonDS input -> libnds keyboard.
X/B/START use Qt::Key bindings in [Instance0.Keyboard]; typing uses synthetic
mouse press/release at DS-pixel coordinates, never a mocked keyboard callback.
XTest global pointer injection was unreliable under GNOME/XWayland.

DS touch coordinate facts (verified against BlocksDS libnds source,
source/arm9/keyboard.c SimpleKbdLower): the default keyboard is a
32x5 grid of 8x16 px cells on the touchscreen, i.e. y in [112,192).
Screen layout "Natural" (ScreenLayout=0) stacks the top screen over
the touchscreen.

Window-mapping facts learned the hard way: Qt creates extra tiny
("Qt Selection Owner", 1x1) windows that also match a name search --
candidates are filtered to MAPPED windows >= 100x100. The melonDS
client window contains a ~34 px menu bar above the render widget, so
the DS content box is NOT simply the centered 2:3 rect of the client.
Instead of guessing the menu height, the layout is CALIBRATED from the
screen itself whenever the touch keyboard is visible: the keyboard is
the bright strip flush against the window's bottom edge (the
touchscreen's bottom edge == client bottom; melonDS has no status
bar), its height is exactly 80 DS px, which yields the scale and box
directly. Falls back to the centered 2:3 assumption only before the
first calibration (boot blank).

XGetImage note: the XImage.data field is c_void_p, NOT c_char_p --
ctypes converts c_char_p fields to bytes truncated at the first NUL.
Read it with ctypes.string_at(data, bytes_per_line*height).

Subcommands print machine-readable results; exit nonzero on failure.
"""

import ctypes
import sys
import time

libX11 = ctypes.CDLL("libX11.so.6")

Display_p = ctypes.c_void_p
Window = ctypes.c_ulong
CurrentTime = 0
AllPlanes = ctypes.c_ulong(~0)
ZPixmap = 2
IsViewable = 2

XOpenDisplay = libX11.XOpenDisplay
XOpenDisplay.restype = Display_p
XOpenDisplay.argtypes = [ctypes.c_char_p]

XDefaultRootWindow = libX11.XDefaultRootWindow
XDefaultRootWindow.restype = Window
XDefaultRootWindow.argtypes = [Display_p]

XSync = libX11.XSync
XSync.argtypes = [Display_p, ctypes.c_int]

XQueryTree = libX11.XQueryTree
XQueryTree.restype = ctypes.c_int
XQueryTree.argtypes = [Display_p, Window, ctypes.POINTER(Window),
                       ctypes.POINTER(Window), ctypes.POINTER(ctypes.POINTER(Window)),
                       ctypes.POINTER(ctypes.c_uint)]

XFetchName = libX11.XFetchName
XFetchName.restype = ctypes.c_int
XFetchName.argtypes = [Display_p, Window, ctypes.POINTER(ctypes.c_char_p)]

XFree = libX11.XFree
XFree.argtypes = [ctypes.c_void_p]


class XWindowAttributes(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
                ("visual", ctypes.c_void_p), ("root", Window),
                ("class", ctypes.c_int), ("bit_gravity", ctypes.c_int),
                ("win_gravity", ctypes.c_int), ("backing_store", ctypes.c_int),
                ("backing_planes", ctypes.c_ulong),
                ("backing_pixel", ctypes.c_ulong),
                ("save_under", ctypes.c_int), ("colormap", ctypes.c_ulong),
                ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int),
                ("all_event_masks", ctypes.c_long), ("your_event_mask", ctypes.c_long),
                ("do_not_propagate_mask", ctypes.c_long),
                ("override_redirect", ctypes.c_int), ("screen", ctypes.c_void_p)]

# Xlib.h ABI: omitting fields lets XGetWindowAttributes overwrite Python heap.
assert ctypes.sizeof(XWindowAttributes) == (136 if ctypes.sizeof(ctypes.c_void_p) == 8 else 92)


XGetWindowAttributes = libX11.XGetWindowAttributes
XGetWindowAttributes.restype = ctypes.c_int
XGetWindowAttributes.argtypes = [Display_p, Window, ctypes.POINTER(XWindowAttributes)]

XTranslateCoordinates = libX11.XTranslateCoordinates
XTranslateCoordinates.restype = ctypes.c_int
XTranslateCoordinates.argtypes = [Display_p, Window, Window, ctypes.c_int,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(Window)]

XSetInputFocus = libX11.XSetInputFocus
XSetInputFocus.restype = ctypes.c_int
XSetInputFocus.argtypes = [Display_p, Window, ctypes.c_int, ctypes.c_ulong]

XGetInputFocus = libX11.XGetInputFocus
XGetInputFocus.restype = ctypes.c_int
XGetInputFocus.argtypes = [Display_p, ctypes.POINTER(Window), ctypes.POINTER(ctypes.c_int)]

XRaiseWindow = libX11.XRaiseWindow
XRaiseWindow.restype = ctypes.c_int
XRaiseWindow.argtypes = [Display_p, Window]

XStringToKeysym = libX11.XStringToKeysym
XStringToKeysym.restype = ctypes.c_ulong
XStringToKeysym.argtypes = [ctypes.c_char_p]

XKeysymToKeycode = libX11.XKeysymToKeycode
XKeysymToKeycode.restype = ctypes.c_uint
XKeysymToKeycode.argtypes = [Display_p, ctypes.c_ulong]

XGetImage = libX11.XGetImage
XGetImage.restype = ctypes.c_void_p
XGetImage.argtypes = [Display_p, Window, ctypes.c_int, ctypes.c_int,
                      ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]

XDestroyImage = libX11.XDestroyImage
XDestroyImage.restype = ctypes.c_int
XDestroyImage.argtypes = [ctypes.c_void_p]

class XInputEvent(ctypes.Structure):
    # XButtonEvent and XKeyEvent have the same ABI; code is button/keycode.
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong),
                ("send_event", ctypes.c_int), ("display", Display_p),
                ("window", Window), ("root", Window), ("subwindow", Window),
                ("time", ctypes.c_ulong), ("x", ctypes.c_int), ("y", ctypes.c_int),
                ("x_root", ctypes.c_int), ("y_root", ctypes.c_int),
                ("state", ctypes.c_uint), ("code", ctypes.c_uint),
                ("same_screen", ctypes.c_int)]


class XEvent(ctypes.Union):
    _fields_ = [("input", XInputEvent), ("pad", ctypes.c_long * 24)]


XSendEvent = libX11.XSendEvent
XSendEvent.restype = ctypes.c_int
XSendEvent.argtypes = [Display_p, Window, ctypes.c_int, ctypes.c_long, ctypes.POINTER(XEvent)]


class Conn:
    def __init__(self):
        self.dpy = XOpenDisplay(None)
        if not self.dpy:
            sys.exit("ds_ui: cannot open X display (need DISPLAY/XWayland)")
        self.root = XDefaultRootWindow(self.dpy)
        self._layout = {}  # window id -> (W, H, ox, oy, s, source)

    # -- windows ------------------------------------------------------------
    def _children(self, w):
        rret, pret, kids, n = Window(), Window(), ctypes.POINTER(Window)(), ctypes.c_uint(0)
        if XQueryTree(self.dpy, w, ctypes.byref(rret), ctypes.byref(pret),
                      ctypes.byref(kids), ctypes.byref(n)) == 0:
            return []
        out = []
        for i in range(n.value):
            out.append(kids[i])
        XFree(kids)
        return out

    def find_windows(self, substr):
        """Windows (recursive from root) whose WM_NAME contains substr."""
        found = []

        def walk(w):
            name = ctypes.c_char_p()
            if XFetchName(self.dpy, w, ctypes.byref(name)) != 0 and name.value:
                if substr.encode() in name.value:
                    found.append((w, name.value.decode("utf-8", "replace")))
                XFree(name)
            for c in self._children(w):
                walk(c)

        walk(self.root)
        return found

    def geometry(self, w):
        """(abs_x, abs_y, width, height, map_state) of window w."""
        wa = XWindowAttributes()
        if XGetWindowAttributes(self.dpy, w, ctypes.byref(wa)) == 0:
            raise RuntimeError("XGetWindowAttributes failed")
        rx, ry = ctypes.c_int(0), ctypes.c_int(0)
        child = Window()
        XTranslateCoordinates(self.dpy, w, self.root, 0, 0,
                              ctypes.byref(rx), ctypes.byref(ry), ctypes.byref(child))
        return rx.value, ry.value, wa.width, wa.height, wa.map_state

    def app_window(self, substr):
        """The real melonDS client window: name matches, is MAPPED, and is
        at least 100x100 (skips Qt's 1x1 'Qt Selection Owner' etc.)."""
        cands = []
        for w, name in self.find_windows(substr):
            x, y, ww, wh, ms = self.geometry(w)
            # GNOME also names its decorated/shadow frame after the client.
            # The real Qt client is the mapped leaf, not that larger frame.
            if ms == IsViewable and ww >= 100 and wh >= 100 and not self._children(w):
                cands.append((ww * wh, w, name, (x, y, ww, wh)))
        if not cands:
            return None
        cands.sort()
        _, w, name, geo = cands[-1]  # largest mapped window wins
        return w, name, geo

    # -- screen -------------------------------------------------------------
    def grab(self, w):
        """Full-window capture. Returns (raw, W, H, stride); raw is BGRA
        bytes, row-major, stride bytes per row."""
        wa = XWindowAttributes()
        if XGetWindowAttributes(self.dpy, w, ctypes.byref(wa)) != 1:
            raise RuntimeError("XGetWindowAttributes failed")
        img = XGetImage(self.dpy, w, 0, 0, wa.width, wa.height, AllPlanes, ZPixmap)
        if not img:
            raise RuntimeError("XGetImage failed")

        class XI(ctypes.Structure):
            _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int),
                        ("xoffset", ctypes.c_int), ("format", ctypes.c_int),
                        ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
                        ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int),
                        ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
                        ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int)]

        xi = XI.from_address(img)
        if xi.bits_per_pixel != 4 * 8:
            XDestroyImage(img)
            raise RuntimeError(f"unexpected bpp {xi.bits_per_pixel}")
        n = xi.bytes_per_line * xi.height
        raw = ctypes.string_at(xi.data, n)  # data is c_void_p; never c_char_p
        W, H, stride = xi.width, xi.height, xi.bytes_per_line
        XDestroyImage(img)
        return raw, W, H, stride

    @staticmethod
    def _px(raw, stride, x, y):
        i = y * stride + x * 4  # ZPixmap little-endian BGRA
        return raw[i + 2], raw[i + 1], raw[i]

    def _box_frac(self, raw, W, H, stride, x0, y0, x1, y1, step=2):
        """Fraction of 'lit' pixels (any channel > 40) in a window-px rect."""
        on = tot = 0
        for y in range(max(0, int(y0)), min(H, int(y1))):
            for x in range(max(0, int(x0)), min(W, int(x1)), step):
                r, g, b = self._px(raw, stride, x, y)
                tot += 1
                if max(r, g, b) > 40:
                    on += 1
        return on / max(tot, 1)

    def _probe_layout(self, w, raw, W, H, stride):
        """Calibrate the DS content box from the touch keyboard: the
        keyboard is the bright strip flush with the window's bottom edge,
        80 DS px tall. Returns (ox, oy, s) in window px, or None."""
        # In the harness's default-size Natural layout the native Qt client
        # has menu chrome at the top (256x406 here), with a 256x384 viewport
        # flush to its bottom. Key-row borders are dark, so counting only a
        # contiguous bright strip does not measure keyboard height reliably.
        s = W / 256.0
        oy = H - 384 * s
        if 0 <= oy <= 64 * s and self._box_frac(
                raw, W, H, stride, 0, oy + 304 * s, W, H) > 0.35:
            return 0, oy, s
        kbd_rows = 0
        while kbd_rows < H:
            y = H - 1 - kbd_rows
            if self._box_frac(raw, W, H, stride, 0, y, W, y + 1, step=3) > 0.4:
                kbd_rows += 1
            else:
                break
        if kbd_rows < 30:  # nothing keyboard-like at the bottom
            return None
        s = kbd_rows / 80.0
        if 384 * s > H + 2 or 256 * s > W + 2:
            return None  # ran past the window: not a keyboard strip
        ox = (W - 256 * s) / 2.0
        oy = H - 384 * s
        return ox, oy, s

    def content_box(self, w):
        """DS content box (ox, oy, s) in WINDOW-relative px, cached per
        window/geometry. Source: 'kbd' (calibrated from the on-screen
        keyboard) or 'guess' (centered 2:3 fallback, pre-calibration)."""
        raw, W, H, stride = self.grab(w)
        lay = self._layout.get(w)
        if lay and lay[0] == W and lay[1] == H and lay[5] == "kbd":
            return lay[2], lay[3], lay[4], lay[5], raw, W, H, stride
        probed = self._probe_layout(w, raw, W, H, stride)
        if probed:
            ox, oy, s = probed
            self._layout[w] = (W, H, ox, oy, s, "kbd")
        else:
            s = min(W / 256.0, H / 384.0)
            ox, oy = (W - 256 * s) / 2.0, (H - 384 * s) / 2.0
            self._layout[w] = (W, H, ox, oy, s, "guess")
        lay = self._layout[w]
        return lay[2], lay[3], lay[4], lay[5], raw, W, H, stride

    def state(self, w):
        """Screen metrics + coarse phase classification.

        editor: libnds touch keyboard visible (bright strip across the
                bottom of the touchscreen).
        text:   keyboard hidden, top screen shows console text.
        blank:  mostly black everywhere (boot/transition).
        """
        ox, oy, s, src, raw, W, H, stride = self.content_box(w)
        top = self._box_frac(raw, W, H, stride, ox, oy, ox + 256 * s, oy + 192 * s)
        bottom = self._box_frac(raw, W, H, stride, ox, oy + 192 * s, ox + 256 * s, oy + 384 * s)
        kbd = self._box_frac(raw, W, H, stride, ox, oy + (192 + 112) * s,
                             ox + 256 * s, oy + 384 * s)
        if kbd > 0.35:
            phase = "editor"
        elif top > 0.008:
            phase = "text"
        else:
            phase = "blank"
        return {"phase": phase, "top_on": round(top, 4), "bottom_on": round(bottom, 4),
                "kbd_on": round(kbd, 4), "layout": src, "scale": round(s, 3),
                "w": W, "h": H}

    def wait_phase(self, w, want, timeout_s, poll_s=0.5):
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            last = self.state(w)
            if last["phase"] == want:
                return last
            time.sleep(poll_s)
        raise TimeoutError(f"wait_phase {want}: last={last}")

    def snap_ppm(self, w, path):
        """Save the DS content box (256x384 DS px region) as binary PPM."""
        ox, oy, s, src, raw, W, H, stride = self.content_box(w)
        with open(path, "wb") as f:
            f.write(f"P6\n256 384\n255\n".encode())
            for dy in range(384):
                y = int(oy + dy * s)
                y = max(0, min(H - 1, y))
                row = bytearray()
                for dx in range(256):
                    x = max(0, min(W - 1, int(ox + dx * s)))
                    r, g, b = self._px(raw, stride, x, y)
                    row += bytes((r, g, b))
                f.write(row)
        return path

    # -- input --------------------------------------------------------------
    def send_input(self, w, kind, code, x=0, y=0, state=0):
        """Target the actual Qt client, not GNOME's global XTest pointer.

        XTest pointer taps can be swallowed by XWayland/compositor focus.
        These events still traverse Qt -> melonDS touch -> DS keyboard code.
        """
        root_x, root_y, *_ = self.geometry(w)
        event = XEvent()
        e = event.input
        e.type, e.send_event, e.display = kind, 1, self.dpy
        e.window, e.root, e.same_screen = w, self.root, 1
        e.x, e.y, e.x_root, e.y_root = x, y, root_x + x, root_y + y
        e.code, e.state = code, state
        mask = {2: 1, 3: 2, 4: 4, 5: 8}[kind]
        if not XSendEvent(self.dpy, w, False, mask, ctypes.byref(event)):
            raise RuntimeError("XSendEvent failed")
        XSync(self.dpy, False)

    def key(self, keysym_name, press_ms=100):
        """Press/release a key on the explicitly focused emulator client."""
        sym = XStringToKeysym(keysym_name.encode())
        if sym == 0:
            raise RuntimeError(f"unknown keysym {keysym_name}")
        code = XKeysymToKeycode(self.dpy, sym)
        if code == 0:
            raise RuntimeError(f"no keycode for {keysym_name}")
        self.send_input(self._target, 2, code)
        time.sleep(press_ms / 1000.0)
        self.send_input(self._target, 3, code)

    def focus(self, w):
        XRaiseWindow(self.dpy, w)
        XSetInputFocus(self.dpy, w, 1, CurrentTime)  # RevertToPointerRoot
        XSync(self.dpy, False)
        got = Window()
        rev = ctypes.c_int(0)
        XGetInputFocus(self.dpy, ctypes.byref(got), ctypes.byref(rev))
        if got.value != w:
            raise RuntimeError(f"focus refused: 0x{got.value:x} != 0x{w:x}")
        self._target = w

    def tap(self, w, ds_x, ds_y, hold_ms=90, settle_s=0.25):
        """Tap DS touchscreen coords on window w."""
        ox, oy, s, src, *_ = self.content_box(w)
        px = int(ox + ds_x * s)
        py = int(oy + (192 + ds_y) * s)
        self.send_input(w, 4, 1, px, py)
        time.sleep(hold_ms / 1000.0)
        self.send_input(w, 5, 1, px, py, state=256)
        time.sleep(settle_s)


# libnds default keyboard touch coords (SimpleKbdLower, verified against
# reference/blocksds-libnds source/arm9/keyboard.c): 32x5 grid of 8x16
# cells starting at touchscreen y=112; a 16-px key spans 2 cells.
KBD = {
    "enter":  (236, 152),  # DVK_ENTER: cells 27-31 of the home row
    "a":      (48,  152),  # 'a': cells 5-6 of the home row
    "bs":     (236, 120),  # DVK_BACKSPACE: cells 27-31 of the digit row
}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, arg = sys.argv[1], sys.argv[2]
    c = Conn()
    if cmd == "find":
        hit = c.app_window(arg)
        if not hit:
            return
        w, name, (x, y, ww, wh) = hit
        print(f"{w} {x} {y} {ww} {wh} {name}")
        return
    w = int(arg, 0)
    if cmd == "focus":
        c.focus(w)
        print("focused")
    elif cmd == "key":
        c.focus(w)
        c.key(sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 40)
        print("key")
    elif cmd == "tap":
        c.focus(w)
        c.tap(w, int(sys.argv[3]), int(sys.argv[4]))
        print("tap")
    elif cmd == "tapkey":
        c.focus(w)
        c.tap(w, *KBD[sys.argv[3]])
        print("tapkey", sys.argv[3])
    elif cmd == "state":
        import json
        print(json.dumps(c.state(w)))
    elif cmd == "wait":
        import json
        print(json.dumps(c.wait_phase(w, sys.argv[3], float(sys.argv[4]))))
    elif cmd == "snap":
        c.snap_ppm(w, sys.argv[3])
        print("snap", sys.argv[3])
    else:
        sys.exit(f"ds_ui: unknown command {cmd}")


if __name__ == "__main__":
    main()
