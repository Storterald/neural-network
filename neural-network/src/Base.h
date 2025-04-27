#pragma once

#if !defined(NDEBUG) && !defined(DEBUG_MODE_ENABLED)
#define DEBUG_MODE_ENABLED
#endif // !NDEBUG && !DEBUG_MODE_ENABLED

#define NN_BEGIN namespace nn {
#define NN_END }
#define NN ::nn::
#define USE_NN using namespace nn;

NN_BEGIN

constexpr float EPSILON       = 1e-8f;
constexpr float LEARNING_RATE = 5e-5f;
constexpr float CLIP_EPSILON  = 0.2f;
constexpr float GAMMA         = 0.99f;
constexpr float LAMBDA        = 0.95f;

NN_END
