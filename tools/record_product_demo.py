#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record real scripted touch input on the clean ROM, offscreen only.

Requires a clean KB3 kit, mtools, Flatpak melonDS, and ffmpeg. No screenshot is
redrawn or synthesized: PPM conversion only expands DS BGR555 pixels to RGB.
"""
import argparse
import json
from pathlib import Path
import struct

import engine_golden as golden

ROOT = Path(__file__).resolve().parent.parent


def ppm(raw, output):
    pixels = struct.unpack('<98304H', Path(raw).read_bytes())
    rgb = bytearray()
    for value in pixels:
        rgb.extend(((value & 31) * 255 // 31, ((value >> 5) & 31) * 255 // 31,
                    ((value >> 10) & 31) * 255 // 31))
    Path(output).write_bytes(b'P6\n256 384\n255\n' + rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/demo.gif')
    args = parser.parse_args()
    kit, work = args.kit.resolve(), args.work.resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be new or empty')
    work.mkdir(parents=True, exist_ok=True)
    model, tokenizer, kb = [kit / ('chatds/' + name + '.bin') for name in ('model', 'tok', 'kb')]
    rom = kit / 'chatds.nds'
    questions = [('fact', 'what is the capital of france', 'paris.'),
                 ('math', 'whats 38 times 47', 'calc(38*47) = 1786')]
    # Reject poor answers before recording anything. These are semantic content
    # checks, not just parity with a possibly bad frozen answer.
    for name, question, answer in questions:
        host = work / ('host-' + name)
        host.mkdir()
        (host / 'run.txt').write_text(f'steps=64\nkbd=0\ninstruct=1\nprompt={question}\n')
        golden.command([ROOT / 'clean/product/build/chatds-host', model, tokenizer, kb,
                        host / 'run.txt', str(host) + '/'], timeout=120)
        actual = golden.parse_result((host / 'out.txt').read_text(), (host / 'ids.txt').read_text(),
                                     (host / 'ctx.txt').read_text())
        shown = (host / 'calc.txt').read_text().strip().removeprefix('calc=') or actual['text'].strip()
        if shown != answer or actual['stop'] != 'eos':
            raise RuntimeError(f'unsuitable demo answer for {question!r}: {shown!r}')
    script = work / 'ui-script.txt'
    script.write_text('wait 24\ntype what is the capital of france\nbutton send\nuntil idle 512\n'
                      'dump fact-answer\nwait 72\ntype whats 38 times 47\nbutton send\n'
                      'until idle 512\ndump math-answer\nwait 72\n')
    device = work / 'device'
    actual = golden.run_rom(rom, model, tokenizer, kb,
                            {'question': '', 'steps': 64, 'instruct': True}, device, 180,
                            card_dir='chatds', script=script)
    if actual['text'].strip() != 'calc(38*47)' or actual['stop'] != 'eos':
        raise RuntimeError('scripted device answer did not match preflight')
    for name in ('frames.txt', 'fact-answer.txt', 'math-answer.txt', 'calc.txt'):
        golden.command(['mcopy', '-o', '-i', device / 'sd.img', '::/chatds/' + name, work / name],
                       capture_output=True)
    for name, _, answer in questions:
        state = (work / (name + '-answer.txt')).read_text()
        if answer not in state or 'busy=0' not in state:
            raise RuntimeError('answer not readable in captured UI state: ' + name)
    frames = work / 'frames'
    frames.mkdir()
    golden.command(['mcopy', '-o', '-i', device / 'sd.img', '::/chatds/*.raw', str(frames) + '/'],
                   capture_output=True)
    concat = []
    for line in (work / 'frames.txt').read_text().splitlines():
        filename, _ = line.split()
        path = frames / filename
        target = path.with_suffix('.ppm')
        ppm(path, target)
        concat += ["file '" + str(target).replace("'", "'\\''") + "'",
                   'duration ' + ('2.5' if 'answer' in filename else '0.12')]
    if not concat:
        raise RuntimeError('no captured frames')
    concat.append(concat[-2])  # concat demuxer needs a final frame for last duration
    (work / 'frames.ffconcat').write_text('\n'.join(concat) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    golden.command(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', work / 'frames.ffconcat',
                    '-filter_complex', 'scale=512:768:flags=neighbor,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                    '-loop', '0', args.output], timeout=120)
    provenance = {'rom_sha256': golden.sha256(rom), 'model_sha256': golden.sha256(model),
                  'tokenizer_sha256': golden.sha256(tokenizer), 'kb_sha256': golden.sha256(kb),
                  'script_sha256': golden.sha256(script), 'gif_sha256': golden.sha256(args.output),
                  'questions': [q for _, q, _ in questions], 'answers': [a for _, _, a in questions],
                  'capture': 'melonDS 1.1 offscreen; real scripted touch input; VRAM BGR555 capture',
                  'playback': 'accelerated, with 2.5-second answer holds; not a latency measurement'}
    golden.write_json(args.output.with_suffix('.json'), provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
