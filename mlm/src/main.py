from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
ctxs = [mx.cpu()] # or, e.g., [mx.gpu(0), mx.gpu(1)]

# MXNet MLMs (use names from mlm.models.SUPPORTED_MLMS)
model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
scorer = MLMScorer(model, vocab, tokenizer, ctxs)
print(scorer.score_sentences(["Hello world!"]))
# >> [-12.410664200782776]
print(scorer.score_sentences(["Hello world!"], per_token=True))

