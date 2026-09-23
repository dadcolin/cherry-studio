import type { AssistantMessage, Context, Model } from '@earendil-works/pi-ai'
import { generateSummary, shouldCompact } from '@earendil-works/pi-coding-agent'
import { describe, expect, it, vi } from 'vitest'

import { resolvePiCompactionSettings } from './piCompactionSettings'

const model: Model<'openai-completions'> = {
  id: 'synthetic-qwen',
  name: 'Synthetic Qwen',
  api: 'openai-completions',
  provider: 'synthetic-provider',
  baseUrl: 'https://example.invalid/v1',
  reasoning: false,
  input: ['text'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 49_152,
  maxTokens: 8_192
}

function response(stopReason: 'error' | 'stop', errorMessage?: string): AssistantMessage {
  return {
    role: 'assistant',
    api: model.api,
    provider: model.provider,
    model: model.id,
    content: [{ type: 'text', text: 'Synthetic summary' }],
    stopReason,
    errorMessage,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
    },
    timestamp: Date.now()
  }
}

describe('Pi compaction on a 49k custom-model context', () => {
  it('starts compaction while there is room for the next request and retains a smaller tail', () => {
    const settings = resolvePiCompactionSettings(model)

    expect(shouldCompact(24_000, model.contextWindow, settings)).toBe(false)
    expect(shouldCompact(26_000, model.contextWindow, settings)).toBe(true)
    expect(settings.keepRecentTokens).toBeLessThan(10_000)
  })

  it('keeps the stock tail on large-context models while reserving their declared output', () => {
    expect(resolvePiCompactionSettings({ contextWindow: 200_000, maxTokens: 8_192 })).toEqual({
      enabled: true,
      reserveTokens: 16_384,
      keepRecentTokens: 20_000
    })
    expect(resolvePiCompactionSettings({ contextWindow: 200_000, maxTokens: 32_768 }).reserveTokens).toBe(32_768)
  })

  it('retries a rejected summary with a smaller prompt while keeping the first and latest context', async () => {
    const promptLengths: number[] = []
    const stream = vi.fn(
      async (_model: Model<'openai-completions'>, context: Context, options: { maxTokens?: number }) => {
        const prompt = (context.messages[0].content as { type: 'text'; text: string }[])[0]
        promptLengths.push(prompt.text.length)
        expect(options.maxTokens).toBeLessThanOrEqual(2_048)
        if (prompt.text.length <= 6_000) {
          expect(prompt.text).toContain('FIRST_SYNTHETIC_ANCHOR')
          expect(prompt.text).toContain('LATEST_SYNTHETIC_ANCHOR')
        }
        return {
          result: async () =>
            prompt.text.length > 6_000 ? response('error', 'maximum context length is 49152 tokens') : response('stop')
        }
      }
    )

    const summary = await generateSummary(
      [
        {
          role: 'user',
          content: `FIRST_SYNTHETIC_ANCHOR\n${'synthetic history '.repeat(8_000)}\nLATEST_SYNTHETIC_ANCHOR`,
          timestamp: Date.now()
        }
      ],
      model,
      16_384,
      'synthetic-key',
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      stream as unknown as Parameters<typeof generateSummary>[9]
    )

    expect(summary).toBe('Synthetic summary')
    expect(promptLengths.length).toBeGreaterThan(1)
    expect(promptLengths.at(-1)).toBeLessThanOrEqual(6_000)
    expect(promptLengths).toEqual([...promptLengths].sort((a, b) => b - a))
  })

  it('does not retry an unrelated provider error', async () => {
    const stream = vi.fn(async () => ({ result: async () => response('error', 'unauthorized') }))

    await expect(
      generateSummary(
        [{ role: 'user', content: 'Synthetic input', timestamp: Date.now() }],
        model,
        16_384,
        'synthetic-key',
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        stream as unknown as Parameters<typeof generateSummary>[9]
      )
    ).rejects.toThrow('unauthorized')
    expect(stream).toHaveBeenCalledTimes(1)
  })
})
