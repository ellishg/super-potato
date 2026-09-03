#include "esp_check.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "utf8proc.h"
#include <math.h>

#define BYTES_TO_KB(bytes) ((uint32_t)((bytes) / 1024))
#define BYTES_TO_MB(bytes) ((uint32_t)((bytes) / (1024 * 1024)))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static const char *TAG = "super-potato";

void heap_alloc_failed_hook(size_t requested_size, uint32_t caps,
                            const char *function_name) {
  ESP_LOGE(TAG,
           "%s failed to allocate %" PRIu32 " KB (free heap size: %" PRIu32
           " KB)",
           function_name, BYTES_TO_KB(requested_size),
           BYTES_TO_KB(esp_get_minimum_free_heap_size()));
}

// Copied from https://github.com/karpathy/llama2.c
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of
                  // multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len;    // max sequence length
} Config;

typedef struct {
  // token embedding table
  const float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  const float *rms_att_weight; // (layer, dim) rmsnorm weights
  const float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  const float *wq; // (layer, dim, n_heads * head_size)
  const float *wk; // (layer, dim, n_kv_heads * head_size)
  const float *wv; // (layer, dim, n_kv_heads * head_size)
  const float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  const float *w1; // (layer, hidden_dim, dim)
  const float *w2; // (layer, dim, hidden_dim)
  const float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  const float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  const float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim,)
  float *xb;     // same, but inside a residual branch (dim,)
  float *xb2;    // an additional buffer just for convenience (dim,)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;      // query (dim,)
  float *k;      // key (dim,)
  float *v;      // value (dim,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->q = calloc(p->dim, sizeof(float));
  s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, const float *ptr,
                        int shared_weights) {
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter
  // counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2;
  // skip what used to be freq_cis_imag (for RoPE)
  ptr += p->seq_len * head_size / 2;
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, const float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++)
    ss += x[j] * x[j];
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++)
    o[j] = weight[j] * (ss * x[j]);
}

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

void matmul(float *xout, float *x, const float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
  // #pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

float *forward(Transformer *transformer, int token, int pos) {
  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  // integer multiplier of the kv sharing in multiquery
  int kv_mul = p->n_heads / p->n_kv_heads;
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  const float *content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim * sizeof(*x));

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {

    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position
    matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
    matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (int v = 0; v < rotn; v++) {
        // the vector to rotate (query or key)
        float *vec = v == 0 ? s->q : s->k;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    // multihead attention. iterate over all heads
    int h;
    // #pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *v =
            s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  float score;
  int length;
  const char *string;
} Vocab;

typedef struct {
  const Vocab *vocab;
  int id;
} TokenIndex;

typedef struct {
  unsigned int max_token_length;
  Vocab *vocabs;
} Tokenizer;

void build_tokenizer(Tokenizer *tokenizer, const void *tokenizer_data,
                     int vocab_size) {
  const uint8_t *data = tokenizer_data;
  memcpy(&tokenizer->max_token_length, data,
         sizeof(tokenizer->max_token_length));
  data += sizeof(tokenizer->max_token_length);
  tokenizer->vocabs = calloc(vocab_size, sizeof(Vocab));

  for (int i = 0; i < vocab_size; i++) {
    memcpy(&tokenizer->vocabs[i].score, data, sizeof(float));
    data += sizeof(float);
    memcpy(&tokenizer->vocabs[i].length, data, sizeof(int));
    data += sizeof(int);
    tokenizer->vocabs[i].string = (const char *)data;
    data += tokenizer->vocabs[i].length;
  }
}

void free_tokenizer(Tokenizer *tokenizer) { free(tokenizer->vocabs); }

int compare_tokens(const void *a, const void *b) {
  const TokenIndex *token_a = (const TokenIndex *)a;
  const TokenIndex *token_b = (const TokenIndex *)b;
  int length = MIN(token_a->vocab->length, token_b->vocab->length);
  int result = memcmp(token_a->vocab->string, token_b->vocab->string, length);
  if (result != 0)
    return result;
  return token_a->vocab->length - token_b->vocab->length;
}

void decode_and_print(const Tokenizer *t, int prev_token, int token) {
  const Vocab *v = &t->vocabs[token];
  int length = v->length;
  const char *piece = v->string;
  // following BOS (1) token, sentencepiece decoder strips any leading
  // whitespace (see PR #89)
  if (prev_token == 1) {
    while (length && piece[0] == ' ') {
      piece++;
      length--;
    }
  }
  if (!length)
    return;

  // Convert tokenizer byte markers such as <0xA5> back to their raw byte.
  if (length == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
      piece[5] == '>') {
    unsigned int byte_value;
    if (sscanf(piece + 3, "%02x", &byte_value) == 1) {
      putchar((unsigned char)byte_value);
      fflush(stdout);
      return;
    }
  }
  printf("%.*s", length, piece);
  fflush(stdout);
}

int str_lookup(char *str, int length, TokenIndex *sorted_vocab,
               int vocab_size) {
  Vocab v;
  v.length = length;
  v.string = str;
  TokenIndex tok;
  tok.vocab = &v;
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex),
                            compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(const Tokenizer *t, TokenIndex *sorted_vocab, int vocab_size,
            char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[]
  // array bos != 0 means prepend the BOS token (=1), eos != 0 means append the
  // EOS token (=2)
  assert(text);

  // create a temporary buffer that will store merge candidates of always two
  // consecutive tokens *2 for concat, +1 for null terminator +2 for UTF8 (in
  // case max_token_length is 1)
  char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default so prepend a dummy prefix token to the
  // input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't
  // have the energy to read more of the sentencepiece code to figure out
  // what it's doing
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", /*length=*/1, sorted_vocab, vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // process the raw (UTF-8) byte sequence of the input string
  for (size_t offset = 0; text[offset] != '\0';) {
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t codepoint_length = utf8proc_iterate(
        (const utf8proc_uint8_t *)text + offset, -1, &codepoint);
    if (codepoint_length < 0) {
      codepoint_length = 1;
    }

    int id =
        str_lookup(text + offset, codepoint_length, sorted_vocab, vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </ s>
      // so the individual bytes only start at index 3
      for (int i = 0; i < codepoint_length; i++) {
        tokens[(*n_tokens)++] = (unsigned char)text[offset + i] + 3;
      }
    }
    offset += codepoint_length;
  }

  // merge the best consecutive pair each iteration, according the scores in
  // vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < (*n_tokens - 1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      const Vocab *v1 = &t->vocabs[tokens[i]];
      const Vocab *v2 = &t->vocabs[tokens[i + 1]];
      size_t merge_len = v1->length + v2->length;
      sprintf(str_buffer, "%.*s%.*s", v1->length, v1->string, v2->length,
              v2->string);
      int id = str_lookup(str_buffer, merge_len, sorted_vocab, vocab_size);
      if (id != -1 && t->vocabs[id].score > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocabs[id].score;
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex,
                float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature,
                   float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++)
      logits[q] /= sampler->temperature;
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp,
                         sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, const Tokenizer *tokenizer,
              TokenIndex *sorted_vocab, Sampler *sampler, char *prompt,
              int steps) {
  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  // +3 for '\0', ?BOS, ?EOS
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
  encode(tokenizer, sorted_vocab, transformer->config.vocab_size, prompt,
         /*bos=*/1, /*eos=*/0, prompt_tokens, &num_prompt_tokens);
  assert(num_prompt_tokens > 0);

  // start the main loop
  // used to time our code, only initialized after first iteration
  long start = 0;
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence
  while (pos < steps) {
    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1)
      break;

    // print the token as string, decode it with the Tokenizer object
    decode_and_print(tokenizer, token, next);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0)
      start = esp_timer_get_time();
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first
  // iteration)
  if (pos > 1) {
    long end = esp_timer_get_time();
    ESP_LOGI(TAG, "achieved tok/s: %f\n",
             (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

void build_transformer(Transformer *transformer, const void *model_data) {
  memcpy(&transformer->config, model_data, sizeof(Config));
  int shared_weights = transformer->config.vocab_size > 0;
  transformer->config.vocab_size = abs(transformer->config.vocab_size);
  memory_map_weights(&transformer->weights, &transformer->config,
                     (const float *)(model_data + sizeof(Config)),
                     shared_weights);
  // allocate the RunState buffers
  malloc_run_state(&transformer->state, &transformer->config);
}

void run_with_model(const void *model_data, const void *tokenizer_data) {
  Transformer transformer;
  build_transformer(&transformer, model_data);

  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_data, transformer.config.vocab_size);
  // Init
  TokenIndex *sorted_vocab =
      malloc(transformer.config.vocab_size * sizeof(TokenIndex));
  for (int i = 0; i < transformer.config.vocab_size; i++) {
    sorted_vocab[i].vocab = &tokenizer.vocabs[i];
    sorted_vocab[i].id = i;
  }
  qsort(sorted_vocab, transformer.config.vocab_size, sizeof(TokenIndex),
        compare_tokens);

  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, /*Temperature=*/1.f,
                /*TopP=*/0.9f, /*Seed=*/101);

  generate(&transformer, &tokenizer, sorted_vocab, &sampler,
           "Tell me a quick story.", MIN(200, transformer.config.seq_len));
  free_sampler(&sampler);
  free_run_state(&transformer.state);
  free(sorted_vocab);
  free_tokenizer(&tokenizer);
}

esp_err_t run(void) {
  ESP_RETURN_ON_ERROR(
      heap_caps_register_failed_alloc_callback(heap_alloc_failed_hook), TAG,
      "Failed to register heap allocation failed callback");
  esp_chip_info_t chip_info;
  esp_chip_info(&chip_info);

  uint32_t flash_size;
  ESP_RETURN_ON_ERROR(esp_flash_get_size(NULL, &flash_size), TAG,
                      "Get flash size failed");

  ESP_LOGI(TAG, "%" PRIu32 "MB %s flash", BYTES_TO_MB(flash_size),
           (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded"
                                                         : "external");
  ESP_LOGI(TAG, "Minimum free heap size: %" PRIu32 " KB",
           BYTES_TO_KB(esp_get_minimum_free_heap_size()));

  const esp_partition_t *partition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
  ESP_RETURN_ON_FALSE(partition, ESP_ERR_NOT_FOUND, TAG,
                      "Partition model not found");

  esp_partition_mmap_handle_t model_map_handle;
  const void *model_data;
  ESP_RETURN_ON_ERROR(
      esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA,
                         (const void **)&model_data, &model_map_handle),
      TAG, "Failed to mmap model");
  size_t model_size = partition->size;
  ESP_LOGI(TAG, "model mmaped at %p with size %" PRIu32 " MB", model_data,
           BYTES_TO_MB(model_size));

  partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                       ESP_PARTITION_SUBTYPE_ANY, "tokenizer");
  ESP_RETURN_ON_FALSE(partition, ESP_ERR_NOT_FOUND, TAG,
                      "Partition tokenizer not found");

  esp_partition_mmap_handle_t tokenizer_map_handle;
  const void *tokenizer_data;
  ESP_RETURN_ON_ERROR(
      esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA,
                         (const void **)&tokenizer_data, &tokenizer_map_handle),
      TAG, "Failed to mmap tokenizer");
  size_t tokenizer_size = partition->size;
  ESP_LOGI(TAG, "tokenizer mmaped at %p with size %" PRIu32 " MB",
           tokenizer_data, BYTES_TO_MB(tokenizer_size));

  run_with_model(model_data, tokenizer_data);

  esp_partition_munmap(model_map_handle);
  esp_partition_munmap(tokenizer_map_handle);

  ESP_LOGI(TAG, "Minimum free heap size: %" PRIu32 " KB",
           BYTES_TO_KB(esp_get_minimum_free_heap_size()));
  return ESP_OK;
}

void app_main(void) {
  esp_err_t err = run();
  if (err != ESP_OK)
    ESP_LOGE(TAG, "Error running application: %s", esp_err_to_name(err));
  ESP_LOGI(TAG, "Application finished");
}