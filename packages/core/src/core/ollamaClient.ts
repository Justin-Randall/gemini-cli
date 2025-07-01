/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

export class OllamaClient implements ContentGenerator {
  constructor(private ollamaUrl: string) {}

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const model = request.model;
    const promptParts: string[] = [];
    for (const content of request.contents as Content[]) {
      if (content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text !== undefined) {
            promptParts.push(part.text);
          }
        }
      }
    }
    const prompt = promptParts.join('\n');

    const response = await fetch(`${this.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();
    const generatedText = data.response;

    return {
      candidates: [
        {
          content: {
            parts: [{ text: generatedText }],
          },
        },
      ],
      // Add other required properties with default or empty values
      text: generatedText,
      functionCalls: [],
      executableCode: '',
      codeExecutionResult: '',
      data: '',
    };
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const model = request.model;
    const promptParts: string[] = [];
    for (const content of request.contents as Content[]) {
      if (content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text !== undefined) {
            promptParts.push(part.text);
          }
        }
      }
    }
    const prompt = promptParts.join('\n');

    const response = await fetch(`${this.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        prompt,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const stream = response.body;
    if (!stream) {
      throw new Error('Response body is null');
    }

    const reader = stream.getReader();
    const decoder = new TextDecoder();

    async function* generate() {
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;
          const parsed = JSON.parse(line);
          const generatedText = parsed.response;
          yield {
            candidates: [
              {
                content: {
                  parts: [{ text: generatedText }],
                },
              },
            ],
            text: generatedText,
            functionCalls: [],
            executableCode: '',
            codeExecutionResult: '',
            data: '',
          };
        }
      }
    }

    return generate();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const model = request.model;
    const promptParts: string[] = [];
    for (const content of request.contents as Content[]) {
      if (content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text !== undefined) {
            promptParts.push(part.text);
          }
        }
      }
    }
    const prompt = promptParts.join('\n');

    const response = await fetch(`${this.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();
    const tokenCount = data.prompt_eval_count;

    return {
      totalTokens: tokenCount,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const model = request.model;
    const promptParts: string[] = [];
    for (const content of request.contents as Content[]) {
      if (content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text !== undefined) {
            promptParts.push(part.text);
          }
        }
      }
    }
    const prompt = promptParts.join('\n');

    const response = await fetch(`${this.ollamaUrl}/api/embed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        input: prompt,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();
    const embeddingValues = data.embeddings[0];

    return {
      embeddings: [
        {
          values: embeddingValues,
        },
      ],
    };
  }
}
