#pragma once

#include <re2/re2.h>
#include "unordered_dense.h"

#include <cassert>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tiktoken {

static auto _byte_pair_merge(
	const std::string &piece,
	const ankerl::unordered_dense::map<std::string, int> &ranks,
	std::function<int (int, int)> func
) -> std::vector<int> {
	std::vector<std::pair<int, int>> parts;
	parts.reserve(piece.size() + 1);
	for (auto idx = 0U; idx < piece.size() + 1; ++idx) {
		parts.emplace_back(idx, std::numeric_limits<int>::max());
	}

	auto get_rank = [&piece, &ranks](
		const std::vector<std::pair<int, int>> &parts,
		int start_idx,
		int skip
	) -> std::optional<int> {
		if (start_idx + skip + 2 < parts.size()) {
			auto s = parts[start_idx].first;
			auto e = parts[start_idx + skip + 2].first;
			auto key = piece.substr(s, e - s);
			auto iter = ranks.find(key);
			if (iter != ranks.end()) {
				return iter->second;
			}
		}
		return std::nullopt;
	};

	for (auto i = 0U; i < parts.size() - 2; ++i) {
		auto rank = get_rank(parts, i, 0);
		if (rank) {
			assert(*rank != std::numeric_limits<int>::max());
			parts[i].second = *rank;
		}
	}

	while (true) {
		if (parts.size() == 1) break;

		auto min_rank = std::make_pair<int, int>(std::numeric_limits<int>::max(), 0);
		for (auto i = 0U; i < parts.size() - 1; ++i) {
			auto rank = parts[i].second;
			if (rank < min_rank.first) {
				min_rank = { rank, i };
			}
		}

		if (min_rank.first != std::numeric_limits<int>::max()) {
			auto i = min_rank.second;
			auto rank = get_rank(parts, i, 1);
			if (rank) {
				parts[i].second = *rank;
			} else {
				parts[i].second = std::numeric_limits<int>::max();
			}
			if (i > 0) {
				auto rank = get_rank(parts, i - 1, 1);
				if (rank) {
					parts[i - 1].second = *rank;
				} else {
					parts[i - 1].second = std::numeric_limits<int>::max();
				}
			}

			parts.erase(parts.begin() + (i + 1));
		} else {
			break;
		}
	}
	std::vector<int> out;
	out.reserve(parts.size() - 1);
	for (auto i = 0U; i < parts.size() - 1; ++i) {
		out.push_back(func(parts[i].first, parts[i + 1].first));
	}
	return out;
}

static auto byte_pair_encode(
	const std::string &piece,
	const ankerl::unordered_dense::map<std::string, int> &ranks
) -> std::vector<int> {
	if (piece.size() == 1) {
		return {ranks.at(piece)};
	}

	auto func = [&piece, &ranks](int start, int stop) -> int {
		std::string key = piece.substr(start, stop - start);
		return ranks.at(key);
	};

	return _byte_pair_merge(piece, ranks, func);
}

class tiktoken {
	public:
    tiktoken() = default;
		tiktoken(
			ankerl::unordered_dense::map<std::string, int> encoder,
			ankerl::unordered_dense::map<std::string, int> special_encoder,
			const std::string &pattern
		) {
			regex_ = std::make_unique<re2::RE2>("(" + pattern + ")");

			std::string special_pattern;
			for (const auto &item : special_encoder) {
				if (!special_pattern.empty()) {
					special_pattern += "|";
				}
				special_pattern += re2::RE2::QuoteMeta(item.first);
			}
			if (special_pattern.empty()) {
				special_regex_ = nullptr;
			} else {
				special_regex_ = std::make_unique<re2::RE2>("(" + special_pattern + ")");
			}

			encoder_ = std::move(encoder);
			special_tokens_encoder = std::move(special_encoder);

			for (const auto &[k, v] : encoder_) {
				decoder_.emplace(v, k);
			}
			assert(encoder_.size() != decoder_.size() && "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?");

			for (const auto &[k, v] : special_tokens_encoder) {
				special_tokens_decoder.emplace(v, k);
			}
		}

    auto encode_ordinary(const std::string &text) -> std::vector<int> {
      return _encode_ordinary_native(text);
    }

		auto encode(const std::string &text) -> std::vector<int> {
			return _encode_native(text, special_tokens_encoder).first;
    }

    auto encode_single_piece(const std::string &text) -> std::vector<int> {
      auto iter = encoder_.find(text);
      if (iter != encoder_.end()) {
        return {iter->second};
      }
      return byte_pair_encode(text, encoder_);
    }

		auto decode(const std::vector<int> &tokens) -> std::string {
			return _decode_native(tokens);
		}

	private:
		template <typename T>
		auto split_with_allowed_special_token(
			re2::StringPiece &input,
			const T &allowed_special
		) -> std::pair<std::optional<std::string>, re2::StringPiece> {
			if (special_regex_ == nullptr) return { std::nullopt, input };

			auto start = input.begin();
			std::string special;
			while (true) {
				if (!re2::RE2::FindAndConsume(&input, *special_regex_, &special)) {
					break;
				}

				if (allowed_special.count(special) == 1) {
					return { std::move(special), re2::StringPiece(start, input.begin() - start - special.size()) };
				}
			}

			return { std::nullopt, input };
		}

    auto _encode_ordinary_native(const std::string &text) -> std::vector<int> {
      std::vector<int> ret;
      re2::StringPiece input(text);

      std::string piece;
      while (re2::RE2::FindAndConsume(&input, *regex_, &piece)) {
        auto iter = encoder_.find(piece);
        if (iter != encoder_.end()) {
          ret.push_back(iter->second);
          continue;
        }
        auto tokens = byte_pair_encode(piece, encoder_);
        ret.insert(ret.end(), tokens.begin(), tokens.end());
      }
      return ret;
    }

		auto _encode_native(
			const std::string &text,
			const ankerl::unordered_dense::map<std::string, int> &allowed_special
		) -> std::pair<std::vector<int>, int> {
			std::vector<int> ret;
			int last_piece_token_len = 0;
			re2::StringPiece input(text);

			while (true) {
				auto [special, sub_input] = split_with_allowed_special_token(input, allowed_special);
				std::string piece;
				while (re2::RE2::FindAndConsume(&sub_input, *regex_, &piece)) {
					auto iter = encoder_.find(piece);
					if (iter != encoder_.end()) {
						last_piece_token_len = 1;
						ret.push_back(iter->second);
						continue;
					}
					auto tokens = byte_pair_encode(piece, encoder_);
					last_piece_token_len = tokens.size();
					ret.insert(ret.end(), tokens.begin(), tokens.end());
				}

				if (special) {
					int token = special_tokens_encoder.at(*special);
					ret.push_back(token);
					last_piece_token_len = 0;
				} else {
					break;
				}
			}

			return { ret, last_piece_token_len };
		}

		auto _decode_native(const std::vector<int> &tokens) -> std::string {
			std::string ret;
			ret.reserve(tokens.size() * 2);
			for (auto token : tokens) {
				std::string token_bytes;
				auto iter = decoder_.find(token);
				if (iter != decoder_.end()) {
					token_bytes = iter->second;
				} else {
					iter = special_tokens_decoder.find(token);
					if (iter != special_tokens_decoder.end()) {
						token_bytes = iter->second;
					} else {
						throw std::runtime_error("unknown token: " + std::to_string(token));
					}
				}
				ret += token_bytes;
			}
			return ret;
		}

		ankerl::unordered_dense::map<std::string, int> encoder_;
		ankerl::unordered_dense::map<std::string, int> special_tokens_encoder;
		ankerl::unordered_dense::map<int, std::string> decoder_;
		ankerl::unordered_dense::map<int, std::string> special_tokens_decoder;
		std::unique_ptr<re2::RE2> regex_;
		std::unique_ptr<re2::RE2> special_regex_;
};

} // namespace tiktoken
