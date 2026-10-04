"""Poll protocol tests without starting melonDS. Run before real emulator tests."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

LIB = Path(__file__).resolve().with_name("melon-lib.sh")


class PollTest(unittest.TestCase):
    def check_poll(self, final, want, expected='OK'):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "test.img"
            image.touch()
            out = Path(tmp) / "out.txt"
            script = r'''
set -euo pipefail
source "$1"
melon_prepare_image_io() { :; }
melon_kill() { :; }
sleep() { :; }
melon_pids() { echo 123; }
# Replace the launcher only, retaining its log activation check.
flatpak() { echo "[melon-image] unbuffered: $image"; }
image=$2
out=$3
final=$4
expected=$5
calls=0
mcopy() {
    [[ $1 == -o ]] || { echo 'missing overwrite flag' >&2; return 1; }
    calls=$((calls + 1))
    if (( calls == 1 )); then
        if [[ $expected == FAIL ]]; then
            printf 'previous answer\nstatus=OK\n' > "$out"
        else
            printf 'answer contains:\nstatus=OK\ngenerated=1\nms=3\n' > "$out"
        fi
    else
        # Wait for background fake launcher to finish writing its ack.
        wait
        printf 'generated=1\n%s\n' "$final" > "$out"
    fi
}
melon_launch_and_poll unused-rom "$image" chatds/out.txt "$out" 3 1 "$expected"
'''
            result = subprocess.run(["bash", "-c", script, "test", str(LIB), str(image),
                                     str(out), final, expected], capture_output=True, text=True, check=True,
                                    env=dict(os.environ, MELON_EMU_LOG=str(Path(tmp) / 'emu.log'),
                                             MELON_MCOPY_LOG=str(Path(tmp) / 'mcopy.log')))
            self.assertEqual(result.stdout.strip(), want, result.stderr)
            # Must not accept the first copy's embedded status=OK.
            self.assertEqual(out.read_text().splitlines()[-1], final)

    def test_partial_result_is_overwritten_until_final_ok(self):
        self.check_poll("status=OK", "1")

    def test_final_failure_is_not_hidden_by_answer_status_text(self):
        self.check_poll("status=FAIL", "0")

    def test_fault_test_waits_for_fail_not_previous_ok(self):
        self.check_poll("status=FAIL", "1", expected='FAIL')


if __name__ == "__main__":
    unittest.main()
