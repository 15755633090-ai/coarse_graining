"""Check launcher routing/provenance without starting property training."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from .launch_paper_pair import main,checkpoint_info

def main_check():
    parent=Path(__file__).parent/'runs';parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        root=Path(directory);checkpoint=root/'best.pt'
        ck={'epoch':7,'valid_loss':.5,'smoke':False,'config':{'structure_dim':12},
            'structure_supervision':{'weight':.1,'definition':{'patterns':[('carbonyl','[#6]=[#8]')]}}}
        torch.save(ck,checkpoint)
        (root/'history.json').write_text(json.dumps([{'epoch':6,'valid_loss':.6},{'epoch':7,'valid_loss':.5}]))
        info=checkpoint_info(checkpoint)
        assert info['checkpoint_epoch']==7 and info['structure_supervision']['weight']==.1
        with patch('motif.launch_paper_pair.subprocess.run') as launch:
            output=root/'default'
            main(['--checkpoint',str(checkpoint),'--output',str(output)])
            launch.assert_called_once()
            command=launch.call_args.args[0]
            assert '--worker' in command and str(checkpoint.resolve()) in command
            assert launch.call_args.kwargs['cwd']==output.resolve()/'source'
            saved=json.loads((output/'launch.json').read_text(encoding='utf-8'))
            assert saved['checkpoint_info']==info
            try:main(['--checkpoint',str(checkpoint),'--output',str(output)])
            except FileExistsError:pass
            else:raise AssertionError('Existing run overwritten')
        with patch('motif.launch_paper_pair.subprocess.run') as launch:
            main(['--checkpoint',str(checkpoint),'--output',str(root/'prepare'),'--prepare-only'])
            launch.assert_not_called()
        ck['epoch']=6;torch.save(ck,checkpoint)
        try:checkpoint_info(checkpoint)
        except ValueError:pass
        else:raise AssertionError('Non-best checkpoint accepted')
    print('PASS: custom checkpoint/epoch, supervision/hash provenance, automatic launch, prepare-only, overwrite and best guards')

if __name__=='__main__':main_check()
