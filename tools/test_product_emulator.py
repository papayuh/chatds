"""Exercise the shared emulator harness protocol without launching an emulator."""
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest import mock

import pytest

import engine_golden as golden


@pytest.mark.parametrize('card_dir,scripted', [('chatds', False), ('chatds', True)])
def test_poll_matches_selected_completion_protocol(tmp_path, card_dir, scripted):
    rom = tmp_path / 'source.nds'
    rom.write_bytes(b'ROM')
    model = tmp_path / 'model.bin'
    model.write_bytes(bytes(36) + struct.pack('<I', 2048))
    script = tmp_path / 'script.txt'
    script.write_text('type q\nbutton send\nuntil idle 10\n')
    work = tmp_path / 'boot'
    trailer = 'tokens=1\nprompt_tokens=1\ngenerated=1\nstop=budget\nms=1\nstatus=OK\n'
    files = {'out.txt': 'answer\n' + trailer, 'ids.txt': '3\n', 'ctx.txt': 'ctx=\n',
             'ui-result.txt': 'frames=1\nstatus=OK\n'}
    polled = []

    def transfer(args, **kwargs):
        source = str(args[-2])
        prefix = '::/' + card_dir + '/'
        assert source.startswith(prefix)
        name = source[len(prefix):]
        Path(args[-1]).write_text(files[name])
        polled.append(name)
        return SimpleNamespace(returncode=0)

    def command(args, **kwargs):
        if str(args[0]) == 'mcopy':
            return transfer(args, **kwargs)
        return SimpleNamespace(returncode=0)

    def launch(args, **kwargs):
        kwargs['stdout'].write(f'[melon-image] unbuffered: {work / "sd.img"}\n')
        kwargs['stdout'].flush()
        return mock.Mock()

    with mock.patch.object(golden, 'pack_image') as pack, \
            mock.patch.object(golden, 'command', side_effect=command), \
            mock.patch.object(golden.subprocess, 'run', side_effect=transfer), \
            mock.patch.object(golden.subprocess, 'Popen', side_effect=launch), \
            mock.patch.object(golden, 'kill_owned'):
        actual = golden.run_rom(rom, model, 'tok.bin', None, {'question': 'q', 'steps': 1},
                                work, 2, card_dir=card_dir, script=script if scripted else None)
    assert polled[0] == ('ui-result.txt' if scripted else 'out.txt')
    assert actual['ids'] == [3] and actual['text'] == 'answer'
    staged = pack.call_args.args[1]
    assert (card_dir + '/ui-script.txt' in staged) == scripted
    config = dict(line.split('=', 1) for line in (work / 'run.txt').read_text().splitlines())
    assert config['kbd'] == str(int(scripted))


def test_product_context_result_does_not_weaken_strict_goldens():
    out = 'answer\ntokens=3\nprompt_tokens=3\ngenerated=1\nstop=context\nms=1\nstatus=OK\n'
    with pytest.raises(ValueError):
        golden.parse_result(out, '4096\n')
    result = golden.parse_result(out, '4096\n', vocab_size=8192, allow_context=True)
    assert result['stop'] == 'context' and result['ids'] == [4096]
    with pytest.raises(ValueError):
        golden.validate_actual(result, {'steps': 64})
