/* Copyright 2020 Greg Tucker
//
// This file is part of brille.
//
// brille is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// brille is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with brille. If not, see <https://www.gnu.org/licenses/>.            */
#ifndef PERMUTATION_TABLE_HPP_
#define PERMUTATION_TABLE_HPP_

#include <map>
#include <array>
#include <tuple>
#include <vector>
#include <algorithm>

// /*
// For future futher optimisation we might want to use the upper triangular
// matrix to pre-allocate all possible permutation entries in the std::vector;
// There are N(N-1)/2 ordered pairs of vertices (i<j) for N total vertices which is
// still very large for large N [𝒪(0.5 N²)].
// To help with this, we will need to convert (i,j) to a linear index into the
// upper triangular part; which is complicated but already solved
// 	https://stackoverflow.com/a/27088560
// */
// static size_t upper_triangular_ij2k(size_t i, size_t j, size_t n) {
// 	return (n*(n-1)/2) - (n-i)*((n-i)-1)/2 +j -i -1;
// }
// static std::tuple<size_t, size_t> upper_triangular_k2ij(size_t k, size_t n){
// 	size_t i = n - 2 - static_cast<size_t>(std::floor(std::sqrt(-8*k + 4*n*(n-1)-7)/2 - 0.5));
// 	size_t j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
// 	return
// }

class PermutationTable
{
public:
	using int_t = int; // uint_fastN_t or uint_leastN_t?
private:
	size_t IndexSize;
	std::map<size_t,size_t> ijmap;
	std::vector<std::vector<int_t>> permutations;
public:
	PermutationTable(size_t ni, size_t branches): IndexSize(ni) {
		std::vector<int_t> identity(branches);
		std::iota(identity.begin(), identity.end(), 0);
		permutations.push_back(identity);
		ijmap.emplace(0u, 0u);
	};
public:
	std::map<size_t,size_t>::const_iterator find(const size_t i, const size_t j) const {
		auto itr = this->ij2key(i,j);
		return ijmap.find(itr);
	}
	bool has(const size_t i, const size_t j) const {
		auto itr =this->find(i,j);
		return itr != ijmap.end();
	}
	size_t set(const size_t i, const size_t j, const size_t idx){
		size_t key = this->ij2key(i,j);
		auto itr = ijmap.find(key);
		if (itr != ijmap.end()) return itr->second;
		ijmap.emplace(key, idx);
		return idx;
	}
	size_t set(const size_t i, const size_t j, const std::vector<int_t>& v){
		bool contains{false};
		size_t idx{0};
		size_t key = this->ij2key(i,j);
		auto itr = ijmap.find(key);
		// if the key is already present, emplace will not overwrite it anyway
		if (itr != ijmap.end()) return itr->second;
		// the ijmap does not contain key (i,j), so check for permutation duplication
		std::tie(contains, idx) = this->find_permutation(v);
		// add this permutation if it's not present
		if (!contains) permutations.push_back(v);
		// and store the mapping
		ijmap.emplace(key, idx);
		return idx;
	}
	std::vector<int_t> safe_get(const size_t i, const size_t j) const {
		auto itr = this->find(i,j);
		return itr != ijmap.end() ? permutations[itr->second] : std::vector<int_t>();
	}
private:
	size_t ij2key(const size_t i, const size_t j) const { return i==j ? 0u : i*IndexSize+j; }
	//
	std::tuple<bool, size_t> find_permutation(const std::vector<int_t>& v) const {
		size_t N = v.size();
		auto equal_perm = [&](const std::vector<int_t>& p){
			if (p.size() != N) return false;
			for (size_t i=0; i<N; ++i)
				if (p[i] != v[i]) return false;
			return true;
		};
		auto itr = std::find_if(permutations.begin(), permutations.end(), equal_perm);
		// returns (true, found_index) or (false, permutations.size())
		return std::make_tuple(itr != permutations.end(), std::distance(permutations.begin(), itr));
	}

};

#endif
