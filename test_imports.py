#!/usr/bin/env python3

print("Testing imports...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    print(f"✗ torch failed: {e}")

try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__}")
except Exception as e:
    print(f"✗ torchvision failed: {e}")

try:
    from transformers import AutoConfig
    print("✓ transformers AutoConfig")
except Exception as e:
    print(f"✗ transformers AutoConfig failed: {e}")

try:
    from transformers import MistralForCausalLM
    print("✓ transformers MistralForCausalLM")
except Exception as e:
    print(f"✗ transformers MistralForCausalLM failed: {e}")

print("Import test complete.")