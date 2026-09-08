import sys
import unittest
from pathlib import Path
from types import MappingProxyType

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parents[1] / "dial"))
try:
    import blend
except ModuleNotFoundError:
    blend = None


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
ADAPTERS = ("cnn", "fox")


class DialApiContractTests(unittest.TestCase):
    def test_weights_exposes_complementary_named_factors(self):
        self.assertIsNotNone(blend)
        self.assertEqual(blend.weights(0.25), {"cnn": 0.25, "fox": 0.75})


@unittest.skipIf(blend is None, "dial implementation has not been created yet")
class DialCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=128,
        )
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(TARGET_MODULES),
            task_type=TaskType.CAUSAL_LM,
            use_rslora=False,
        )
        cls.model = get_peft_model(LlamaForCausalLM(config), lora_config, adapter_name="cnn")
        cls.model.add_adapter("fox", lora_config)
        cls.model.base_model.set_adapter(list(ADAPTERS))
        cls.layers = tuple(module for module in cls.model.modules() if isinstance(module, LoraLayer))

        with torch.no_grad():
            for layer_index, layer in enumerate(cls.layers, start=1):
                for adapter_index, adapter in enumerate(ADAPTERS, start=1):
                    a_value = 0.01 * (layer_index + adapter_index)
                    b_value = 0.02 * (layer_index + adapter_index)
                    layer.lora_A[adapter].weight.fill_(a_value)
                    layer.lora_B[adapter].weight.fill_(b_value)

        cls.base_scaling = MappingProxyType({
            adapter: MappingProxyType({layer: layer.scaling[adapter] for layer in cls.layers})
            for adapter in ADAPTERS
        })
        cls.pristine_deltas = {
            adapter: {layer: layer.get_delta_weight(adapter).detach().clone() for layer in cls.layers}
            for adapter in ADAPTERS
        }

    def test_fixture_activates_exact_adapter_pair_and_nonzero_deltas(self):
        self.assertEqual(set(self.layers[0].active_adapters), set(ADAPTERS))
        self.assertEqual(len(self.layers), 14)
        for adapter in ADAPTERS:
            for layer in self.layers:
                self.assertTrue(torch.count_nonzero(self.pristine_deltas[adapter][layer]).item())

    def test_set_lambda_endpoints_are_rebased_from_pristine_scales(self):
        blend.set_lambda(self.model, 0.25, self.base_scaling)
        blend.set_lambda(self.model, 1.0, self.base_scaling)

        for layer in self.layers:
            self.assertTrue(torch.allclose(layer.get_delta_weight("cnn"), self.pristine_deltas["cnn"][layer]))
            self.assertTrue(torch.allclose(layer.get_delta_weight("fox"), torch.zeros_like(self.pristine_deltas["fox"][layer])))
            self.assertEqual(layer.scaling["cnn"], self.base_scaling["cnn"][layer])
            self.assertEqual(layer.scaling["fox"], 0.0)

        blend.set_lambda(self.model, 0.0, self.base_scaling)
        for layer in self.layers:
            self.assertTrue(torch.allclose(layer.get_delta_weight("cnn"), torch.zeros_like(self.pristine_deltas["cnn"][layer])))
            self.assertTrue(torch.allclose(layer.get_delta_weight("fox"), self.pristine_deltas["fox"][layer]))
            self.assertEqual(layer.scaling["cnn"], 0.0)
            self.assertEqual(layer.scaling["fox"], self.base_scaling["fox"][layer])


if __name__ == "__main__":
    unittest.main()
