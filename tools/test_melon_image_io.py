"""Host regression for melonDS's buffered-tail symptom and exact-image shim.

Run: python3 -m unittest discover -s tools -p test_melon_image_io.py
The real one-token/rename regression is ds/hwkit/test-publication.sh.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent


class ImageIOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.shim = cls.root / "image-io.so"
        subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared",
                        "-fPIC", str(TOOLS / "melon-image-io.c"), "-ldl",
                        "-o", str(cls.shim)], check=True)
        src = cls.root / "buffered.c"
        src.write_text('''#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    int fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE *f = fdopen(fd, "wb");
    if (!f || fwrite("done", 1, 4, f) != 4) return 2;
    /* Like SIGTERM: no fclose, atexit, or implicit stdio flush. */
    _exit(0);
}
''')
        cls.exe = cls.root / "buffered"
        subprocess.run(["cc", str(src), "-o", str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_writer(self, name, preload=False, target=None):
        path = self.root / name
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env.pop("MELON_UNBUFFERED_IMAGE", None)
        if preload:
            env["LD_PRELOAD"] = str(self.shim)
        if target is not None:
            env["MELON_UNBUFFERED_IMAGE"] = str(target)
        subprocess.run([str(self.exe), str(path)], env=env, check=True,
                       capture_output=True)
        return path.read_bytes()

    def test_unflushed_tail_is_lost_without_shim(self):
        self.assertEqual(self.run_writer("buffered.img"), b"")

    def test_exact_image_visible_even_without_clean_exit(self):
        path = self.root / "image.img"
        self.assertEqual(self.run_writer(path.name, True, path), b"done")

    def test_unrelated_file_remains_buffered(self):
        self.assertEqual(self.run_writer("other.img", True, self.root / "image.img"), b"")

    def test_unset_target_does_not_change_stdio(self):
        self.assertEqual(self.run_writer("unset.img", True), b"")


if __name__ == "__main__":
    unittest.main()
