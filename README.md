# Re-implementation of LMFusion with Qwen3

This repository contains a re-implementation of the paper: [LMFusion: Adapting Pretrained Language Models for Multimodal Generation](https://arxiv.org/abs/2412.15188).

## Overview

LMFusion presents an interesting and powerful architecture for multimodal generation. The original framework adapts pretrained text-only large language models (LLMs) to process both text and images. This is achieved by introducing parallel transformer modules to handle image data while leveraging the existing weights of the language model for text processing. During training, the text-specific modules are frozen to preserve the model's original language capabilities.

## Motivation and Philosophy

The authors of LMFusion detailed the model's structure, training setup, and experimental results in their paper. However, the official code, model weights, and the private dataset used for training were not released.

As is often said, the devil is in the details. Due to constraints of length and narrative flow, academic papers may omit critical implementation details. For instance, specifics about the U-Net's Downsampler and Upsampler architecture or the use of techniques like QK-Norm are not fully described. This lack of available code makes direct replication challenging.

This repository is an attempt to reproduce the paper as faithfully as possible. Where the paper is ambiguous, we will make informed implementation choices and clearly document them. Our goal is to provide a clear and functional codebase for the community.

## Our Implementation: Utilizing Qwen3

While the paper's implementation uses LLaMA-3, this project utilizes the more recent **Qwen3** language model. A key feature of Qwen3 is its unified training approach that combines a "thinking mode" for complex, step-by-step reasoning and a "non-thinking mode" for quick responses.

### Leveraging Qwen3's Thinking Mode

This implementation takes full advantage of Qwen3's "thinking mode." We have specifically adapted the chat template to harness this capability, allowing the model to perform more complex reasoning during multimodal generation. This involves structuring the input to prompt the model to generate intermediate reasoning steps before producing the final output.

## The Development Journey: Insights and Reflections

Building a model of this complexity is not a trivial task. The overall design—structuring the model's architecture, defining how data flows between components, and assigning tasks to specific modules—can be even more challenging than writing the code itself. A solid understanding of both the `transformers` and `diffusers` libraries is essential.

Throughout the process, maintaining a clear distinction between the **prefill** and **decode** stages is critically important for success.

### On Building with AI Assistants

I initially considered whether a large portion of this project could be completed with the help of an advanced LLM like Gemini 2.5 Pro. These models are incredibly powerful for reading and understanding large codebases across multiple files, often pinpointing precise details.

However, the complexity of this project appeared to exceed the model's capability for end-to-end implementation. When a subtle error is introduced and goes unnoticed by both the developer and the AI, it can cascade and compound, leading to a state of confusion. This realization led me to restart and dive deep into the code myself.

My experience has been that if you are unclear about any detail, your code will almost certainly have issues. Simply directing an LLM through a few conversational turns without a deep, personal understanding of the code's behavior is often insufficient to solve the problem and can lead to getting lost. Grasping what the code is doing at every step is key to ensuring correctness.

That said, the programming capabilities of LLMs are advancing at a breathtaking pace. Who knows what will be possible in six months, a year, or even five years?

## Target Audience

This repository is intended for:
1.  Researchers and developers who wish to reproduce LMFusion.
2.  Anyone interested in a deep dive into developing custom LLMs or MLLMs using the `transformers` library.

We all come from different backgrounds. To provide a thorough guide to building LMFusion, it's necessary to touch upon the architecture of general-purpose LLMs like Qwen3 and the intricacies of the `transformers` library. This creates a balancing act: some may find the content too basic, while others might feel key prerequisites are missing.

As the reader, you are the most important part of this equation. Please feel free to skip over familiar sections and focus on the parts that will contribute most to your growth. Your time and energy are valuable.

## Development Status and Future Maintenance

Please note that this project is currently under development. The code in this repository should be considered a work in progress and an interim release. We will continue to actively develop this project and incorporate future improvements.