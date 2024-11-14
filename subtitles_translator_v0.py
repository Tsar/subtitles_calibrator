#!/usr/bin/env python3.8

import sys
import json
import copy
import asyncio

import pysrt

from async_openai_requests import log, requestChatCompletion, StatusNot200Exception

SUBS_FALLBACK_ENCODING = 'cp1251'

SUBS_BATCH_SIZE_TO_TRANSLATE = 50
SUBS_NEXT_BATCH_OVERLAP = 0

PROMPT_HEADER = '''
Below is a JSON with subtitles for a part of %s.
Translate them to %s language please.
Provide the response in the following format:
{
  "translated_subtitles": [
    "translation for the first",
    "translation for the second",
    ...
  ]
}
Elements of resulting translated_subtitles object should correspond to elements of original_subtitles. 
'''.strip()

TRANSLATED_SUBTITLES_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "translated_subtitles",
        "schema": {
            "type": "object",
            "properties": {
                "translated_subtitles": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["translated_subtitles"],
            "additionalProperties": False
        }
    }
}

with open('openai_creds.json', 'r') as openaiCreds:
    OPENAI_API_KEY = json.load(openaiCreds)['apiKey']

def readSubs(filename):
    try:
        return pysrt.open(filename)
    except UnicodeDecodeError:
        return pysrt.open(filename, encoding=SUBS_FALLBACK_ENCODING)

async def translateSubtitles(contentDescription, targetLanguage, inputSrtFilename, outputSrtFilename):
    subs = readSubs(inputSrtFilename)
    result = copy.deepcopy(subs)
    pos = 0

    while pos < len(subs):
        prompt = PROMPT_HEADER % (contentDescription, targetLanguage) + '\n\n' + json.dumps(
            {"original_subtitles": list(map(lambda sub: sub.text, subs[pos:pos + SUBS_BATCH_SIZE_TO_TRANSLATE]))},
            indent=2,
        )
        prompt = prompt.strip()

        log(f'Making request to GPT to translate subs range {pos}..{pos + SUBS_BATCH_SIZE_TO_TRANSLATE} of {len(subs)}')

        try:
            response = await requestChatCompletion(
                messages=[{'role': 'user', 'content': prompt}],
                gptModel='gpt-4o',
                apiKey=OPENAI_API_KEY,
                additionalParams={"response_format": TRANSLATED_SUBTITLES_RESPONSE_FORMAT},
            )
        except StatusNot200Exception as ex:
            log(f'Translating subtitles failed: {ex}, {ex.detailsForLogging}')
            break

        translated = json.loads(response)['translated_subtitles']
        if len(translated) == 0:
            log('GPT returned no translations, finishing')
            break

        for i, newSub in enumerate(translated, start=pos):
            result[i].text = newSub

        pos += len(translated) - SUBS_NEXT_BATCH_OVERLAP

    result.save(outputSrtFilename, encoding='utf-8')
    log(f'Wrote "{outputSrtFilename}"')

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f'Usage: {0} <content description, for example: series Castle, season 1> <target language> <existing subtitles filename> <output filename>')
        sys.exit(1)

    asyncio.run(translateSubtitles(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
