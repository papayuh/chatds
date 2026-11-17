"""Exercise actual atomic publisher, including injected flush/close/rename errors."""
import ctypes as C
import errno
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def io(tmp_path_factory):
    work = tmp_path_factory.mktemp('io-errors')
    source = work / 'probe.c'
    source.write_text('''
#include <errno.h>
#include "plumbing.h"
int fail_stage;
/* Only metadata engine callbacks are irrelevant to these filesystem tests. */
void chatds_session_get_info(const chatds_session *s, chatds_session_info *out) { (void)s; (void)out; }
const char *cds_stop_string(cds_stop stop) { (void)stop; return "none"; }
int __real_rename(const char *, const char *);
int __wrap_rename(const char *a, const char *b) {
    if (fail_stage == 1) { errno = EACCES; return -1; }
    return __real_rename(a, b);
}
int __real_fflush(FILE *);
int __wrap_fflush(FILE *f) {
    if (fail_stage == 2) { errno = ENOSPC; return -1; }
    return __real_fflush(f);
}
int __real_fclose(FILE *);
int __wrap_fclose(FILE *f) {
    int rc = __real_fclose(f);
    if (fail_stage == 3) { errno = EIO; return -1; }
    return rc;
}
int publish(const char *dir, int stage) {
    fail_stage = stage;
    cds_answer answer = {0};
    cds_result result = {.answer = &answer, .text = "answer", .ok = 1, .instruct = 1};
    return cds_publish(dir, &result);
}
''')
    so = work / 'probe.so'
    subprocess.run(['cc', '-std=c99', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I' + str(ROOT / 'clean/include'), '-I' + str(ROOT / 'clean/plumbing'),
                    source, ROOT / 'clean/plumbing/io.c', '-Wl,--wrap=rename,--wrap=fflush,--wrap=fclose',
                    '-o', so], check=True)
    lib = C.CDLL(str(so), use_errno=True)
    lib.publish.argtypes = [C.c_char_p, C.c_int]
    lib.cds_atomic_open.argtypes = [C.c_char_p]
    lib.cds_atomic_open.restype = C.c_void_p
    lib.cds_atomic_commit.argtypes = [C.c_void_p, C.c_char_p]
    return lib


@pytest.mark.parametrize('stage,message,code', [(1, 'result rename failed', errno.EACCES),
                                               (2, 'result write failed', errno.ENOSPC),
                                               (3, 'result write failed', errno.EIO)])
def test_publication_io_failures_are_not_success(io, tmp_path, capfd, stage, message, code):
    assert io.publish(str(tmp_path).encode() + b'/', stage) == -1
    diagnostic = capfd.readouterr().err
    assert message in diagnostic and f'errno={code}' in diagnostic
    assert str(tmp_path / 'out.txt') in diagnostic
    assert not (tmp_path / 'out.txt').exists()
    assert io.publish(str(tmp_path).encode() + b'/', 0) == 0
    assert (tmp_path / 'out.txt').read_text().endswith('status=OK\n')
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('length', [251, 252, 256])
def test_atomic_suffix_bounds(io, tmp_path, length, capfd):
    # Valid parent and exact full path length; no nonexistent-directory confound.
    parent = str(tmp_path) + '/'
    path = parent + 'x' * (length - len(parent))
    f = io.cds_atomic_open(path.encode())
    if length == 251:
        assert f
        assert io.cds_atomic_commit(f, path.encode()) == 0
        assert Path(path).exists()
    else:
        assert not f and C.get_errno() == errno.ENAMETOOLONG
        assert 'result path failed' in capfd.readouterr().err
        assert not list(tmp_path.iterdir())
