#pragma once

#include "IPredictor.hpp"
#include "Shared.hpp"

struct ErrorInfo {
  uint32_t data[2], sum, mask, collected;

  void reset() {
    memset(this, 0, sizeof(*this));
  }
};

class Mixer : protected IPredictor {
protected:
  static constexpr int MAX_LEARNING_RATE = 8 * 65536 - 1;
  static constexpr int MIN_LEARNING_RATE_S1 = 2 * 65536 - 1;
  static constexpr int MIN_LEARNING_RATE_SN = 6 * 65536 - 1;

  const Shared * const shared;
  const uint32_t n; /**< max inputs */
  const uint32_t m; /**< max contexts */
  const uint32_t s; /**< max context sets */
  const int lowerLimitOfLearningRate; /**< for linear learning rate decay */
  const bool isAdaptiveLearningRate; /**< linked to command line option '-a' */
  int scaleFactor; /**< scale factor for dot product */
  Array<short, 64> tx; /**< n inputs from add() */
  Array<short, 64> wx; /**< n*m weights */
  Array<uint32_t> cxt; /**< s contexts */
  Array<ErrorInfo> info; /**< stats for the adaptive learning rates  */
  Array<int> rates; /**< learning rates */
  uint32_t numContexts {}; /**< number of contexts (0 to s)  */
  uint32_t base {}; /**< offset of next context */
  uint32_t nx {}; /**< number of inputs in tx, 0 to n */
  Array<int> pr; /**< last result (scaled 12 bits) */
public:
  /**
    * Mixer m(n, m, s) combines models using @ref m neural networks with
    * @ref n inputs each, of which up to @ref s may be selected.  If s > 1 then
    * the outputs of these neural networks are combined using another
    * neural network (with arguments s, 1, 1). If s = 1 then the
    * output is direct.
    * @param n
    * @param m
    * @param s
    */
  Mixer(const Shared* sh, int n, int m, int s);

  ~Mixer() override = default;
  /**
    * Returns the output prediction that the next bit is 1 as a 12 bit number (0 to 4095).
    * @return the prediction
    */
  virtual int p() = 0;
  virtual void setScaleFactor(int sf0, int sf1) = 0;
  virtual void promote(int x) = 0;

  /**
    * Input x (call up to n times)
    * m.add(stretch(p)) inputs a prediction from one of n models.  The
    * prediction should be positive to predict a 1 bit, negative for 0,
    * nominally +-256 to +-2K.  The maximum allowed value is +-32K but
    * using such large values may cause overflow if n is large.
    * @param x
    */
  void add(int x);

  /**
    *  Selects @ref cx as one of @ref range neural networks to
    *  use. 0 <= cx < range. Should be called up to @ref s times such
    *  that the total of the ranges is <= @ref m.
    * @param cx
    * @param range
    * @param rate
    */
  void set(uint32_t cx, uint32_t range);
  void reset();
};
