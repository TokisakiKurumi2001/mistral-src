from api import ProcessRequest, Token, ProcessResponse, TokenizeRequest, TokenizeResponse, DecodeRequest, DecodeResponse
from helper import generate_one_seq

import time
import logging
import uvicorn
import torch
import warnings

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from pathlib import Path
from fastapi import FastAPI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

def main(
    model_path: str = "../mistral-7B-v0.1",
    port=8080,
):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

    @app.post("/process")
    async def process_request(input_data: ProcessRequest) -> ProcessResponse:
        logger.info("Using device: {}".format(transformer.device))
        prompts = [input_data.prompt] # using one prompt only
        
        t0 = time.perf_counter()
        generated_words, generated_tokens, logprobs = generate_one_seq(
            prompts=prompts,
            model=transformer,
            tokenizer=tokenizer,
            max_tokens=input_data.max_new_tokens,
            chunk_size=500,
            temperature=input_data.temperature,
            echo_prompt=input_data.echo_prompt,
        )

        t = time.perf_counter() - t0

        tokens_generated = len(generated_words)
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
        )
        output = "".join(generated_words)

        logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        logprobs = logprobs[0]
        tokens = generated_tokens[0].tolist()
        generated_tokens_server = []
        for t, lp in zip(tokens, logprobs):
            generated_tokens_server.append(
                Token(text=tokenizer.decode(t), logprob=lp, top_logprob={'random': 0.7})
            )
        logprobs_sum = sum(logprobs)
        # Process the input data here
        return ProcessResponse(
            text=output, tokens=generated_tokens_server, logprob=logprobs_sum, request_time=t
        )

    @app.post("/tokenize")
    async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
        logger.info("Using device: {}".format(transformer.device))
        t0 = time.perf_counter()
        encoded = tokenizer.encode(input_data.text, bos=True, device=transformer.device)
        t = time.perf_counter() - t0
        tokens = encoded.tolist()
        return TokenizeResponse(tokens=tokens, request_time=t)

    @app.post("/decode")
    async def decode(input_data: DecodeRequest) -> DecodeResponse:
        logger.info("Using device: {}".format(transformer.device))
        t0 = time.perf_counter()
        decoded = tokenizer.processor.decode(input_data.tokens)
        t = time.perf_counter() - t0
        return DecodeResponse(text=decoded, request_time=t)

    uvicorn.run(app, port=port, log_level='info')


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main, as_positional=False)