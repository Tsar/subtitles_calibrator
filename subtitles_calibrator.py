#!/usr/bin/env python3.8
import copy
import sys
import json
import asyncio

import pysrt

from async_openai_requests import log, requestChatCompletion, StatusNot200Exception

SUBS_FALLBACK_ENCODING = 'cp1251'

SUBS_BATCH_SIZE_TO_MATCH = 20
SUBS_NEXT_BATCH_OVERLAP = 2

TO_BE_DELETED_MARKER = '###TO_BE_DELETED###'

PROMPT_HEADER = '''
Below are several enumerated sentences, which are part of subtitles for a movie.
Match the phrases from the first list with those from the second list.
In your answer, indicate the correspondence by numbers. Ignore any extra phrases.
One phrase from the first list can correspond to multiple phrases from the second list, and vice versa.
Example of response:
{
  "corresponding_subtitles": [
    [1, 3],
    [1, 4],
    [2, 5],
    [3, 7],
    [4, 7]
  ]
}
First element in the integer array must be number of phrase in the first list, second element must be number of
phrase in the second list.
'''.strip()

CORRESPONDING_SUBTITLES_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "corresponding_subtitles",
        "schema": {
            "type": "object",
            "properties": {
                "corresponding_subtitles": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                }
            },
            "required": ["corresponding_subtitles"],
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

async def matchAndFixSubtitles(correctSrtFilename, badTimingsSrtFilename):
    subs1 = readSubs(correctSrtFilename)
    subs2 = readSubs(badTimingsSrtFilename)
    result = copy.deepcopy(subs1)
    pos1 = 0
    pos2 = 0

    while pos1 < len(subs1) and pos2 < len(subs2):
        prompt = PROMPT_HEADER + '\n\n=== List 1 ===\n\n'
        for i, sub in enumerate(subs1[pos1:pos1 + SUBS_BATCH_SIZE_TO_MATCH], start=1):
            prompt += f'{i}\n{sub.text}\n\n'
        prompt += '\n=== List 2 ===\n\n'
        for i, sub in enumerate(subs2[pos2:pos2 + SUBS_BATCH_SIZE_TO_MATCH], start=1):
            prompt += f'{i}\n{sub.text}\n\n'
        prompt = prompt.strip()

        log(
            f'Making request to GPT to match subs1 range {pos1}..{pos1 + SUBS_BATCH_SIZE_TO_MATCH} of {len(subs1)} ' +
            f'with subs2 range {pos2}..{pos2 + SUBS_BATCH_SIZE_TO_MATCH} of {len(subs2)}'
        )

        try:
            response = await requestChatCompletion(
                messages=[{'role': 'user', 'content': prompt}],
                gptModel='gpt-4o',
                apiKey=OPENAI_API_KEY,
                additionalParams={"response_format": CORRESPONDING_SUBTITLES_RESPONSE_FORMAT},
            )
        except StatusNot200Exception as ex:
            log(f'Subtitles matching failed: {ex}, {ex.detailsForLogging}')
            break

        corresponding = json.loads(response)['corresponding_subtitles']
        if len(corresponding) == 0:
            log('GPT returned no matches, finishing')
            break

        id1ToIds2 = {}
        for corr in corresponding:
            if corr[0] in id1ToIds2:
                id1ToIds2[corr[0]].append(corr[1])
            else:
                id1ToIds2[corr[0]] = [corr[1]]
        id2ToIds1 = {}
        for corr in corresponding:
            if corr[1] in id2ToIds1:
                id2ToIds1[corr[1]].append(corr[0])
            else:
                id2ToIds1[corr[1]] = [corr[0]]

        for id1 in id1ToIds2:
            ids2 = id1ToIds2[id1]
            ids2.sort()
            newSub = ''
            for id2 in ids2:
                newSub += subs2[pos2 + id2 - 1].text + ' '
            result[pos1 + id1 - 1].text = newSub.strip()

        for id2 in id2ToIds1:
            ids1 = id2ToIds1[id2]
            if len(ids1) > 1:
                ids1.sort()
                result[pos1 + ids1[0] - 1].end = result[pos1 + ids1[-1] - 1].end
                for id1 in ids1[1:]:
                    result[pos1 + id1 - 1].text = TO_BE_DELETED_MARKER

        maxId1 = max(map(lambda x: x[0], corresponding))
        maxId2 = max(map(lambda x: x[1], corresponding))
        pos1 += max(maxId1 - SUBS_NEXT_BATCH_OVERLAP, 1)
        pos2 += max(maxId2 - SUBS_NEXT_BATCH_OVERLAP, 1)

    for i in range(len(result) - 1, -1, -1):
        if result[i].text == TO_BE_DELETED_MARKER:
            del result[i]
    newSubsFilename = badTimingsSrtFilename[:-4] + '.autofixed.srt'
    result.save(newSubsFilename, encoding='utf-8')
    log(f'Wrote "{newSubsFilename}"')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {0} <SRT filename with correct timings> <other language SRT filename with incorrect timings>')
        sys.exit(1)

    asyncio.run(matchAndFixSubtitles(sys.argv[1], sys.argv[2]))
