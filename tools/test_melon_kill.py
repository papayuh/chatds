"""Kill only the owned emulator for this ROM, not a harness mentioning its path."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LIB = Path(__file__).with_name('melon-lib.sh').resolve()


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux /proc matcher')
class KillTest(unittest.TestCase):
    def test_exact_emulator_argv_not_shell_or_other_rom(self):
        code = ('import ctypes,sys,time; '
                'ctypes.CDLL(None).prctl(15,sys.argv[1].encode(),0,0,0); '
                'print("READY",flush=True); time.sleep(30)')
        with tempfile.TemporaryDirectory() as td:
            rom = str(Path(td) / 'model with spaces.nds')
            processes = [subprocess.Popen([sys.executable,'-c',code,name,arg],
                                         stdout=subprocess.PIPE,text=True)
                         for name,arg in [('melonDS',rom),('harness',rom),('melonDS',rom+'.other')]]
            try:
                for process in processes:
                    self.assertEqual(process.stdout.readline().strip(),'READY')
                # This shell itself also has the full ROM in its argv.
                result = subprocess.run(['bash','-c','source "$1"; melon_kill "$2"',
                                         'harness',str(LIB),rom],timeout=12)
                self.assertEqual(result.returncode,0,'matcher killed its caller')
                self.assertIsNotNone(processes[0].poll())
                self.assertIsNone(processes[1].poll(),'killed a non-emulator mentioning ROM')
                self.assertIsNone(processes[2].poll(),'matched a different ROM prefix')
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)
                    process.stdout.close()


if __name__ == '__main__':
    unittest.main()
