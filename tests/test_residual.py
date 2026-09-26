from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.model import MultiscaleMolecularModel, MultiscaleModelConfig
from multiscale_tokenizer import training as t
from test_training import _TinyPropertyDataset


class ResidualTests(TestCase):
    def source(self, directory):
        base = MultiscaleMolecularModel(DiffusionEncoder(ModelConfig(hidden_dim=8)),
            config=MultiscaleModelConfig(hidden_dim=8, experiment_mode="baseline_stage2_frozen"))
        scaler = t.TargetScaler.fit(_TinyPropertyDataset(directory, split="train"))
        path = Path(directory) / "base.pt"
        torch.save(dict(schema_version=4, experiment_mode="baseline_stage2_frozen",
            task_spec=asdict(t.TASK_SPECS["lipo"]), model_config=asdict(base.config),
            model_state=base.state_dict(), best_epoch=0, model_state_epoch=0,
            history=[dict(epoch=0, valid_rmse=1.0)], target_scaler=asdict(scaler),
            provenance={"dataset": "fixture"}), path)
        return path, base

    def test_zero_init_update_and_reload(self):
        with TemporaryDirectory() as directory, patch.object(t, "load_encoder", side_effect=lambda **kw: DiffusionEncoder(ModelConfig(hidden_dim=8))):
            path, base = self.source(directory)
            base.eval()
            x = torch.zeros(2, 4, 5, dtype=torch.long)
            x[..., 0] = 6
            bonds = torch.zeros(2, 4, 4, dtype=torch.long)
            bonds[:, 0, 1] = bonds[:, 1, 0] = 1
            mask = torch.ones(2, 4, dtype=torch.bool)
            expected = base(x, bonds, mask).prediction
            for mode in ("baseline_residual_frozen", "multiscale_residual_frozen"):
                model = t.build_property_model(spec=t.TASK_SPECS["lipo"], experiment_mode=mode,
                    model_seed=0, partition_seed=100000, dropout=0.1, device=torch.device("cpu"), base_init_checkpoint=path)
                model.train()
                self.assertFalse(model.base_head.training)
                self.assertFalse(model.encoder.training)
                actual = model(x,bonds,mask).prediction
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                before = {k:v.clone() for k,v in model.state_dict().items() if k.startswith(("encoder.","base_head."))}
                optimizer = t.build_optimizer(model, encoder_learning_rate=0, downstream_learning_rate=1e-3, weight_decay=1e-5)
                (actual-expected.detach()-1).square().mean().backward()
                self.assertGreater(model.prediction_head[-1].weight.grad.abs().sum().item(),0)
                optimizer.step()
                for k,v in before.items():
                    torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
                self.assertTrue(all(p.grad is None for p in model.base_head.parameters()))
                save = Path(directory)/"residual.pt"
                torch.save(dict(schema_version=4,model_config=asdict(model.config),model_state=model.state_dict(),task_spec=asdict(t.TASK_SPECS["lipo"])),save)
                restored,_,_ = t.load_property_model(save)
                model.eval()
                torch.testing.assert_close(model(x,bonds,mask).prediction,restored(x,bonds,mask).prediction,rtol=0,atol=0)

    def test_initial_candidate_retained_when_training_does_not_improve(self):
        with TemporaryDirectory() as directory, patch.object(t,"load_encoder",side_effect=lambda **kw: DiffusionEncoder(ModelConfig(hidden_dim=8))), patch.object(t,"OGBMoleculePropertyDataset",_TinyPropertyDataset), patch.object(t,"_experiment_provenance",return_value={"dataset":"fixture"}):
            source,_=self.source(directory)
            def metrics(*args, **kwargs):
                return dict(loss=1.0,rmse=1.0,mae=0.5,r2=0.0)
            with patch.object(t,"_run_epoch",side_effect=metrics):
                result=t.train_property_model(directory,task="lipo",output_dir=Path(directory)/"run",device="cpu",epochs=1,experiment_mode="baseline_residual_frozen",base_init_checkpoint=source)
            self.assertEqual(result["best_epoch"],-1)
            self.assertEqual(result["model_state_epoch"],-1)
            self.assertEqual(result["current_epoch"],0)
            self.assertEqual([r["epoch"] for r in result["history"]],[-1,0])
            self.assertEqual(result["model_state"]["prediction_head.3.weight"].count_nonzero(),0)
