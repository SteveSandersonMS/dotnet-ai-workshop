# .NET AI Workshop

Welcome! This workshop aims to help you get started adding AI features to .NET applications.

Why would you want to? Many, perhaps most, nontrivial applications can be made more powerful and productive through adding AI-based features. It can automate many business processes and let users complete their tasks faster while remaining in control.

These features include **classification**, **summarization**, **data extraction/cleaning**, **anomaly detection**, **translation**, **sentiment detection**, **semantic search**, **Q&A chatbots**, **voice assistants**, and more.

## Who is this relevant to?

This is relevant to **developers getting started with AI in .NET applications**.

We will assume you're already familiar with .NET and C#, including typical app patterns such as dependency injection (DI). You *don't* need to know Blazor or web programming, though a couple of the examples will use them. In most cases, examples will be plain console apps to preserve the focus on AI topics.

**You don't need prior knowledge of AI technology.** We'll focus on the **usage** of AI services, including LLMs and embeddings, to implement app features such as those listed above. This includes understanding capabilities, limitations, problems, and risks. We'll focus a lot on optimizing for reliability.

We won't be creating or training new AI models from scratch, so it doesn't matter whether you know how things like tokenizers or transformers work.

## What do I need?

You'll need:

- A development environment, including:
  - .NET 9
  - A code editor (e.g., VS 2022, VS Code with the C# Dev Kit extension, or Rider)
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/)

As for AI services, you can use any of the following:

| Service | Notes |
| --- | --- |
| OpenAI platform | Ideal. Instructions provided. |
| Azure OpenAI | Ideal. Instructions provided. |
| Ollama (local) | Instructions provided, but some things are trickier to make work well, and it's nowhere near as fast |
| Others (Google Gemini, Anthropic, HuggingFace, etc) | Instructions are not provided but the code should work with slight adaptations, since we use [Microsoft.Extensions.AI.Abstractions](https://aka.ms/m.e.ai) which supports arbitrary backends.

Chapter 1 contains more information about how to set these up.

## Contents

Jump in wherever you like. If you're new to this, starting at chapter 1 makes sense.

| Index | Title | Contents | Topics |
| ---| --- | --- | --- |
| 1 | [Build a Quiz App](./instructions/1_BuildAQuizApp.md) | Get started quickly with a casual exercise. This helps build familiarity with using LLMs without needing too much theory. | Chat completion. Prompt engineering. Prompt injection. |
| 2 | [Embeddings](./instructions/2_Embeddings.md) | Learn how to compute the semantic similarity of text and perform semantic search. This is a basic building block for many AI features. | Embeddings. Semantic search. |
| 3 | [Vector Search](./instructions/3_VectorSearch.md) | Build an index of embeddings for high-performance semantic search. | Vector databases. FAISS. |
| 4 | [Language Models (Part 1)](./instructions/4_LanguageModels_Part1.md) | Use LLMs to implement a range of application features. | Completion. Streaming. Structured output.
| 5 | [Language Models (Part 2)](./instructions/5_LanguageModels_Part2.md) | Use LLMs to implement a range of application features. | Function calling. Chatbots. Middleware.
| 6 | [Retrieval Augmented Generation](./instructions/6_RAGChatbot.md) | Implement an end-to-end Q&A chatbot, from data ingestion to quality evaluation. | Ingestion. Chunking. Embedding. Retrieval. Citations. Evaluation. RAG triad. |
| 7 | [Vision](./instructions/7_Vision.md) | Combine multi-modal models with earlier topics to build an image monitoring system. | Multi-modality. Structured output. Small models. Caching. |
| 8 | [Realtime](./instructions/8_Realtime.md) | Produce responses in realtime using an audio-to-audio model, creating a voice assistant. | Realtime APIs |
