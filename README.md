# Re-implementation of LMFusion with Qwen3

This repository contains a re-implementation of the paper: [LMFusion: Adapting Pretrained Language Models for Multimodal Generation](https://arxiv.org/abs/2412.15188).

## Overview

The original LMFusion framework adapts pretrained text-only large language models (LLMs) for multimodal generation, enabling them to process both text and images. This is achieved by introducing parallel transformer modules to handle image data while leveraging the existing weights of the language model for text processing. During training, the text-specific modules are frozen to preserve the model's original language capabilities.

## Our Implementation: Utilizing Qwen3

While the paper's implementation uses LLaMA-3, this project utilizes the more recent **Qwen3** language model. A key feature of Qwen3 is its unified training approach that combines a "thinking mode" for complex, step-by-step reasoning and a "non-thinking mode" for quick responses. (However, it is worth noting that more recent releases of Qwen models appear to be moving away from this unified approach.)

### Leveraging Qwen3's Thinking Mode

This implementation takes full advantage of Qwen3's "thinking mode." We have specifically adapted the chat template to harness this capability, allowing the model to perform more complex reasoning during multimodal generation. This involves structuring the input to prompt the model to generate intermediate reasoning steps before producing the final output.

## Future Maintenance

This project will be actively maintained to incorporate future updates and improvements.