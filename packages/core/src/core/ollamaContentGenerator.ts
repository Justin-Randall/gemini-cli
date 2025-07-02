/**
 * @license
 * Copyright 2025 Justin Randall and Playscale PTE LTD
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
import { ContentGeneratorConfig } from './contentGenerator.js';

interface OllamaMessage {
  role: string;
  content: string;
}

interface OllamaMessageResponse {
  role: string;
  content: string;
  thinking?: string; // Optional field for thinking messages
  tool_calls?: Array<{
    id: string;
    type: string;
    function: {
      name: string;
      arguments: string;
    };
  }>;
}
interface OllamaChatRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  think: boolean;
}

interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: OllamaMessageResponse;
  done: boolean;
}

interface OllamaModelDetails {
  parent_model: string;
  format: string;
  family: string;
  families: string[];
}

interface OllamaModelTag {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: OllamaModelDetails;
  parameter_size: number;
  quantization_level: string;
}

interface OllamaModelTagResponse {
  models: OllamaModelTag[];
}

interface OllamaModelInfo {
  'general.architecture': string;
  'general.basename': string;
  'general.file_type': number;
  'general.license': string;
  'general.parameter_count': number;
  'general.quantization_version': number;
  'general.size_label': string;
  'general.type': string;
  'qwen3.attention.head_count': number;
  'qwen3.attention.head_count_kv': number;
  'qwen3.attention.key_length': number;
  'qwen3.attention.layer_norm_rms_epsilon': number;
  'qwen3.attention.value_length': number;
  'qwen3.block_count': number;
  'qwen3.context_length': number;
  'qwen3.embedding_length': number;
  'qwen3.feed_forward_length': number;
  'qwen3.rope.freq_base': number;
  'tokenizer.ggml.add_bos_token': boolean;
  'tokenizer.ggml.bos_token_id': number;
  'tokenizer.ggml.eos_token_id': number;
  'tokenizer.ggml.merges': null;
  'tokenizer.ggml.model': string;
  'tokenizer.ggml.padding_token_id': number;
  'tokenizer.ggml.pre': string;
  'tokenizer.ggml.token_type': null;
  'tokenizer.ggml.tokens': null;
}

interface OllamaModelTensor {
  name: string;
  type: string;
  shape: number[];
}

interface OllamaFullModelInfo {
  license: string;
  details: OllamaModelDetails;
  model_info: OllamaModelInfo;
  tensors: OllamaModelTensor[];
  capabilities: string[];
}

interface OllamaModel {
  ollamaModelTag: OllamaModelTag;
  ollamaFullModelInfo: OllamaFullModelInfo;
}

export class OllamaContentGenerator implements ContentGenerator {
  constructor(private contentGeneratorConfig: ContentGeneratorConfig) {}

  // when selecting a model, if it is not mapped, request
  // model information from the Ollama api endpoint.
  private ollamaModelMap: Map<string, Promise<OllamaModel>> = new Map();
  private tagsPromise: Promise<OllamaModelTag[]> | null = null;

  private getTags(): Promise<OllamaModelTag[]> {
    if (!this.tagsPromise) {
      this.tagsPromise = fetch(
        `${this.contentGeneratorConfig.ollamaUrl}/api/tags`,
      )
        .then((res) => {
          if (!res.ok) {
            throw new Error(
              `GET ${this.contentGeneratorConfig.ollamaUrl}/api/tags failed: ${res.status}`,
            );
          }
          return res.json() as Promise<OllamaModelTagResponse>;
        })
        .then((json) => {
          if (!json.models || !Array.isArray(json.models)) {
            throw new Error(`Unexpected /api/tags response shape`);
          }
          return json.models;
        });
    }
    return this.tagsPromise;
  }

  private fetchShow(modelName: string): Promise<OllamaFullModelInfo> {
    return fetch(`${this.contentGeneratorConfig.ollamaUrl}/api/show`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName }),
    }).then((res) => {
      if (!res.ok) {
        throw new Error(
          `POST ${this.contentGeneratorConfig.ollamaUrl}/api/show failed for ${modelName}: ${res.status}`,
        );
      }
      return res.json() as Promise<OllamaFullModelInfo>;
    });
  }

  private getModel(modelName: string): Promise<OllamaModel> {
    // 1) cache hit?
    if (this.ollamaModelMap.has(modelName)) {
      return this.ollamaModelMap.get(modelName)!;
    }

    // 2) otherwise kick off the lookup & store its promise:
    const modelPromise = (async (): Promise<OllamaModel> => {
      // a) wait for tags, find the tag entry
      const tags = await this.getTags();
      const tag = tags.find((t) => t.model === modelName);
      if (!tag) {
        throw new Error(
          `Model not found in ${this.contentGeneratorConfig.ollamaUrl}/api/tags: ${modelName}`,
        );
      }

      // b) fetch the full info
      const fullInfo = await this.fetchShow(modelName);

      // c) combine into one object
      return {
        ollamaModelTag: tag,
        ollamaFullModelInfo: fullInfo,
      };
    })();

    this.ollamaModelMap.set(modelName, modelPromise);
    return modelPromise;
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const model = request.model;

    const ollamaModel = await this.getModel(model);
    if (!ollamaModel) {
      throw new Error(`Model not found: ${model}`);
    }

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

    const response = await fetch(
      `${this.contentGeneratorConfig.ollamaUrl}/api/generate`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
        }),
      },
    );

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

    const ollamaModel = await this.getModel(model);
    if (!ollamaModel) {
      throw new Error(`Model not found: ${model}`);
    }

    // build messages array from request contents
    // we are going to start using the "chat" endppoint, so we aren't going to use the "prompt" field
    // instead, we will use the "messages" field
    const messages: OllamaMessage[] = [];
    for (const content of request.contents as Content[]) {
      if (content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text !== undefined) {
            messages.push({
              role: content.role || 'user',
              content: part.text,
            });
          }
        }
      }
    }
    if (messages.length === 0) {
      throw new Error('No messages provided in request contents');
    }

    // create the request body
    const requestBody: OllamaChatRequest = {
      model,
      messages,
      stream: true,
      think: ollamaModel.ollamaFullModelInfo.capabilities.includes('thinking'),
    };
    const requestBodyJson = JSON.stringify(requestBody);

    const response = await fetch(
      `${this.contentGeneratorConfig.ollamaUrl}/api/chat`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: requestBodyJson,
      },
    );

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
      if (!ollamaModel.ollamaFullModelInfo.capabilities.includes('tools')) {
        // JAR : TODO - if tool use was requested and the model does not support tools,
        // we should yield a response indicating that tools are not supported. Get to this
        // when implementing tool calling.
        // yield a response letting the user know that tools are not supported
        // yield {
        //   candidates: [
        //     {
        //       content: {
        //         role: 'model', // MUST be model, see geminiChat.ts validateHistory()
        //         parts: [{ text: 'Tools were provided, but this model does not support tool use.' }],
        //       },
        //     },
        //   ],
        //   text: 'This model does not support tools.',
        //   functionCalls: [],
        //   executableCode: '',
        //   codeExecutionResult: '',
        //   data: '',
        // };
        // return;
      }
      let buffer = '';
      let finished = false;
      while (!finished) {
        const { done, value } = await reader.read();
        if (done) {
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;
          // const parsed = JSON.parse(line);

          // get the OllamaMessage from the parsed response
          const ollamaChatResponse: OllamaChatResponse = JSON.parse(line);
          // const generatedText = ollamaChatResponse.message.content;
          const content = ollamaChatResponse.message.content ?? '';
          const thinking = ollamaChatResponse.message.thinking ?? '';
          const isThinking = thinking.length > 0 && content.length === 0;
          const messageText = isThinking ? thinking : content;

          yield {
            candidates: [
              {
                content: {
                  role: 'model', // MUST be model, see geminiChat.ts validateHistory()
                  parts: [{ text: messageText, thought: isThinking }],
                },
              },
            ],
            text: messageText,
            functionCalls: [],
            executableCode: '',
            codeExecutionResult: '',
            data: '',
          };

          if (ollamaChatResponse.done) {
            finished = true;
            return;
          }
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

    const response = await fetch(
      `${this.contentGeneratorConfig.ollamaUrl}/api/generate`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
        }),
      },
    );

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

    const response = await fetch(
      `${this.contentGeneratorConfig.ollamaUrl}/api/embed`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          input: prompt,
        }),
      },
    );

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
