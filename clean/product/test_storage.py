"""Launch-root policy and bounded joins, using the actual shared C helpers."""
import ctypes as C
import errno
from pathlib import Path
import subprocess
import shutil

import pytest

from test_product import executable, invoke  # shared real product fixture

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def storage(tmp_path_factory):
    work = tmp_path_factory.mktemp('storage')
    source = work / 'storage.c'
    source.write_text('''
#include "chatds_storage.h"
const char *root(const char *drive) { return cds_storage_root(drive); }
int join(char *out, size_t cap, const char *dir, const char *name) {
    return cds_path_join(out, cap, dir, name);
}
''')
    so = work / 'storage.so'
    subprocess.run(['cc', '-std=c99', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-I' + str(ROOT / 'clean/include'), source, '-o', so], check=True)
    lib = C.CDLL(str(so), use_errno=True)
    lib.root.argtypes = [C.c_char_p]
    lib.root.restype = C.c_char_p
    lib.join.argtypes = [C.c_void_p, C.c_size_t, C.c_char_p, C.c_char_p]
    return lib


@pytest.mark.parametrize('drive,expected', [(b'fat:/', b'fat:/chatds/'), (b'sd:/', b'sd:/chatds/')])
def test_card_root_not_rom_folder(storage, drive, expected):
    assert storage.root(drive) == expected
    for name in (b'run.txt', b'model.bin', b'tok.bin', b'kb.bin', b'kb4.bin',
                 b'round3-act.cdsr', b'round3-op.cdsr', b'round3-tag.cdsr',
                 b'out.txt', b'debug.log', b'session.txt', b'session-results.tsv',
                 b'kernel.txt', b'ui-script.txt', b'frames.txt', b'ui-result.txt', b'f000.raw'):
        out = C.create_string_buffer(256)
        assert storage.join(out, len(out), expected, name) == 0
        assert out.value == expected + name


@pytest.mark.parametrize('drive', [None, b'', b'nand:/', b'nand2:/', b'nitro:/', b'fat:', b'sd:/roms/', b'other:/'])
def test_no_unsafe_drive_fallback(storage, drive):
    assert storage.root(drive) is None
    assert C.get_errno() == errno.ENODEV


@pytest.mark.parametrize('cap', [0, 1, 17, 18, 19])
def test_path_bounds_no_truncation_or_overwrite(storage, cap):
    # sd:/chatds/run.txt = 18 bytes, plus terminator.
    out = C.create_string_buffer(b'!' * 32, 33)
    rc = storage.join(out, cap, b'sd:/chatds/', b'run.txt')
    if cap < 19:
        assert rc == -1 and C.get_errno() == errno.ENAMETOOLONG
        if cap:
            assert out.value == b''
    else:
        assert rc == 0 and out.value == b'sd:/chatds/run.txt'
    assert out.raw[cap:32] == b'!' * (32 - cap)


@pytest.mark.parametrize('drive', [b'fat:/', b'sd:/'])
def test_selected_card_reads_assets_and_publishes_together(storage, executable, tmp_path, monkeypatch, drive):
    monkeypatch.chdir(tmp_path)
    card = Path(storage.root(drive).decode())
    card.mkdir(parents=True)
    for name in ('act', 'op', 'tag'):
        shutil.copyfile(ROOT / f'campaign/router/models/round3-{name}.cdsr', card / f'round3-{name}.cdsr')
    # A learned text-op forces head reads; outputs and debug log stay on this card.
    result, _ = invoke(executable, card, 'steps=2\ninstruct=1\nprompt=sort these words zipper harbor rabbit river\n',
                       output=card, learned_directory=card.resolve())
    assert result.returncode == 0, result.stderr
    assert (card / 'out.txt').read_text().endswith('status=OK\n')
    assert 'learned_text=' in (card / 'debug.log').read_text()
    assert not list(card.glob('*.tmp'))
    other = Path('fat:/chatds' if drive == b'sd:/' else 'sd:/chatds')
    assert not other.exists()


def test_missing_selected_card_does_not_select_other_card(storage, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'fat:/chatds').mkdir(parents=True)
    (tmp_path / 'fat:/chatds/run.txt').write_text('steps=1\n')
    assert storage.root(b'sd:/') == b'sd:/chatds/'
    assert not Path('sd:/chatds/run.txt').exists()
    # Existence/content of either card must not influence the policy.
    (tmp_path / 'sd:/chatds').mkdir(parents=True)
    (tmp_path / 'sd:/chatds/run.txt').write_text('steps=bad\n')
    assert storage.root(b'sd:/') == b'sd:/chatds/'
    assert storage.root(b'fat:/') == b'fat:/chatds/'
