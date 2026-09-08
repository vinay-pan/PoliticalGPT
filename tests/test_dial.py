import asyncio
import random
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
LAMBDA_GRID = [-0.5, 0, 0.25, 0.5, 0.75, 1, 1.5]


class DialApiContractTests(unittest.TestCase):
    def test_weights_exposes_complementary_named_factors(self):
        self.assertIsNotNone(blend)
        self.assertEqual(blend.weights(0.25), {"cnn": 0.25, "fox": 0.75})


@unittest.skipIf(blend is None, "dial implementation has not been created yet")
class DialCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self._reset_fixture()

    def _reset_fixture(self):
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
        self.model = get_peft_model(LlamaForCausalLM(config), lora_config, adapter_name="cnn")
        self.model.add_adapter("fox", lora_config)
        self.model.base_model.set_adapter(list(ADAPTERS))
        self.layers = tuple(module for module in self.model.modules() if isinstance(module, LoraLayer))

        with torch.no_grad():
            for layer_index, layer in enumerate(self.layers, start=1):
                for adapter_index, adapter in enumerate(ADAPTERS, start=1):
                    a_value = 0.01 * (layer_index + adapter_index)
                    b_value = 0.02 * (layer_index + adapter_index)
                    layer.lora_A[adapter].weight.fill_(a_value)
                    layer.lora_B[adapter].weight.fill_(b_value)

        self.base_scaling = MappingProxyType({
            adapter: MappingProxyType({layer: layer.scaling[adapter] for layer in self.layers})
            for adapter in ADAPTERS
        })
        self.pristine_deltas = {
            adapter: {layer: layer.get_delta_weight(adapter).detach().clone() for layer in self.layers}
            for adapter in ADAPTERS
        }

    def _assert_live_formula_and_scales(self, lambda_value):
        expected_factors = blend.weights(lambda_value)
        for layer in self.layers:
            expected_delta = (
                lambda_value * self.pristine_deltas["cnn"][layer]
                + (1.0 - lambda_value) * self.pristine_deltas["fox"][layer]
            )
            live_delta = layer.get_delta_weight("cnn") + layer.get_delta_weight("fox")
            self.assertTrue(torch.allclose(live_delta, expected_delta, atol=1e-5))
            for adapter, factor in expected_factors.items():
                self.assertEqual(layer.scaling[adapter], self.base_scaling[adapter][layer] * factor)

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

    def test_exact_grid_matches_direct_formula_and_fresh_cat_oracle(self):
        for grid_index, lambda_value in enumerate(LAMBDA_GRID):
            self._reset_fixture()
            reference_adapter = f"cat_oracle_grid_{grid_index}"
            self.model.add_weighted_adapter(
                ["cnn", "fox"],
                [lambda_value, 1.0 - lambda_value],
                reference_adapter,
                combination_type="cat",
            )

            blend.set_lambda(self.model, lambda_value, self.base_scaling)
            self._assert_live_formula_and_scales(lambda_value)
            for layer in self.layers:
                expected_delta = (
                    lambda_value * self.pristine_deltas["cnn"][layer]
                    + (1.0 - lambda_value) * self.pristine_deltas["fox"][layer]
                )
                self.assertTrue(torch.allclose(layer.get_delta_weight(reference_adapter), expected_delta, atol=1e-5))

    def test_thousand_random_changes_rebase_scales_without_drift(self):
        random_lambdas = random.Random(20260906)
        for _ in range(1_000):
            lambda_value = random_lambdas.uniform(-0.5, 1.5)
            blend.set_lambda(self.model, lambda_value, self.base_scaling)
            self._assert_live_formula_and_scales(lambda_value)

        for lambda_value in (0.0, 1.0):
            blend.set_lambda(self.model, lambda_value, self.base_scaling)
            self._assert_live_formula_and_scales(lambda_value)


class SerializedGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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
        self.model = get_peft_model(LlamaForCausalLM(config), lora_config, adapter_name="cnn")
        self.model.add_adapter("fox", lora_config)
        self.model.base_model.set_adapter(list(ADAPTERS))
        self.model.eval()
        self.layers = tuple(module for module in self.model.modules() if isinstance(module, LoraLayer))

        with torch.no_grad():
            for layer_index, layer in enumerate(self.layers, start=1):
                for adapter_index, adapter in enumerate(ADAPTERS, start=1):
                    layer.lora_A[adapter].weight.fill_(0.01 * (layer_index + adapter_index))
                    layer.lora_B[adapter].weight.fill_(0.02 * (layer_index + adapter_index))

        self.base_scaling = MappingProxyType({
            adapter: MappingProxyType({layer: layer.scaling[adapter] for layer in self.layers})
            for adapter in ADAPTERS
        })

    async def test_overlapping_endpoint_calls_serialize_scales_and_model_outputs(self):
        self.assertTrue(hasattr(blend, "generate_serialized"))

        lock = asyncio.Lock()
        first_snapshot = asyncio.Event()
        critical_sections = []
        traces = {"fox": [], "cnn": []}
        input_ids = torch.tensor([[1, 2, 3, 4]])

        async def generate_callback(label):
            critical_sections.append(("enter", label))
            for snapshot_index in range(3):
                scales = {
                    adapter: tuple(layer.scaling[adapter] for layer in self.layers)
                    for adapter in ADAPTERS
                }
                with torch.no_grad():
                    logits = self.model(input_ids=input_ids).logits.detach().clone()
                traces[label].append((scales, logits))
                if label == "fox" and snapshot_index == 0:
                    first_snapshot.set()
                await asyncio.sleep(0)
            critical_sections.append(("exit", label))

        fox_task = asyncio.create_task(
            blend.generate_serialized(self.model, 0.0, self.base_scaling, lock, lambda: generate_callback("fox"))
        )
        await first_snapshot.wait()
        cnn_task = asyncio.create_task(
            blend.generate_serialized(self.model, 1.0, self.base_scaling, lock, lambda: generate_callback("cnn"))
        )
        await asyncio.gather(fox_task, cnn_task)

        self.assertEqual(
            critical_sections,
            [("enter", "fox"), ("exit", "fox"), ("enter", "cnn"), ("exit", "cnn")],
        )
        for label, expected_factors in (("fox", {"cnn": 0.0, "fox": 1.0}), ("cnn", {"cnn": 1.0, "fox": 0.0})):
            self.assertEqual(len(traces[label]), 3)
            for scales, logits in traces[label]:
                for adapter, factor in expected_factors.items():
                    expected_scales = tuple(self.base_scaling[adapter][layer] * factor for layer in self.layers)
                    self.assertEqual(scales[adapter], expected_scales)
                self.assertTrue(torch.allclose(logits, traces[label][0][1]))
        self.assertFalse(torch.allclose(traces["fox"][0][1], traces["cnn"][0][1]))


if __name__ == "__main__":
    unittest.main()
