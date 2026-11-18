/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file profiler.hpp
 * @brief A scoped timer that prints how long its scope took.
 */
#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <utility>

namespace rko_lio::core {

/** Simple timing utility that prints elapsed time for a scope or code block. */
struct Timer {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<double>;

  Timer() : label("Execution"), start_time(Clock::now()) {}
  explicit Timer(std::string label) : label(std::move(label)), start_time(Clock::now()) {}
  Timer(const Timer&) = default;
  Timer(Timer&&) = default;
  Timer& operator=(const Timer&) = default;
  Timer& operator=(Timer&&) = default;
  ~Timer() {
    const auto end_time = Clock::now();
    const Duration duration = end_time - start_time;
    std::cout << label << " took " << duration.count() << " seconds.\n";
  }
  std::string label;
  TimePoint start_time;
};

} // namespace rko_lio::core
