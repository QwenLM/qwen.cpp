#pragma once

#include <string>
#include <string_view>

namespace base64 {

static auto pos_of_char(const unsigned char chr) -> size_t {
  if      (chr >= 'A' && chr <= 'Z') return chr - 'A';
  else if (chr >= 'a' && chr <= 'z') return chr - 'a' + ('Z' - 'A')               + 1;
  else if (chr >= '0' && chr <= '9') return chr - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
  else if (chr == '+' || chr == '-') return 62;
  else if (chr == '/' || chr == '_') return 63;
  else throw std::runtime_error("Input is not valid base64-encoded data.");
}

inline auto decode(std::string_view s) -> std::string {
  if (s.empty()) throw std::runtime_error("empty input");
  size_t length = s.length();
  size_t idx = 0;

  std::string out;
  out.reserve(length / 4 * 3);

  while (idx < length) {
    size_t pos_of_char_1 = pos_of_char(s.at(idx + 1));
    out.push_back(static_cast<std::string::value_type>(((pos_of_char(s.at(idx+0))) << 2 ) + ((pos_of_char_1 & 0x30) >> 4)));
    if ((idx + 2 < length) && s.at(idx + 2) != '=' && s.at(idx + 2) != '.') {
      size_t pos_of_char_2 = pos_of_char(s.at(idx + 2));
      out.push_back(static_cast<std::string::value_type>(((pos_of_char_1 & 0x0f) << 4) + ((pos_of_char_2 & 0x3c) >> 2)));
      if ((idx + 3 < length) && s.at(idx + 3) != '=' && s.at(idx + 3) != '.') {
        out.push_back(static_cast<std::string::value_type>(((pos_of_char_2 & 0x03) << 6) + pos_of_char(s.at(idx+3))));
      }
    }
    idx += 4;
  }
  return out;
}

} // namespace base64
