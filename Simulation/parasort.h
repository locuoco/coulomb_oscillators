//  Sort an array in parallel using multiple threads
//  Copyright (C) 2016 Amir Baserinia (baserinia.github.io)

//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

//  modified by Alessandro Lo Cuoco to allow for custom comparators

//  To use this program, simply include it in your project and call 
//  the function "parasort(size, list, comp, np, sf)".
//      size: size of array
//      list: array pointer
//		comp: comparator (default = std::less<T>())
//      np: number of threads (default = 2)
//      sf: sampling factor (default = 100)

#ifndef PARASORT_H
#define PARASORT_H

#include <cstdlib>      // rand
#include <cstring>      // memcpy
#include <vector>       // std::vector
#include <algorithm>    // std::sort
#include <thread>       // std::thread
#include <functional>   // std::less

//------------------------------------------------------------------------------
// Sort an array using std::sort
// sz: length of array
// list: array pointer
// comp: comparator
template <typename T, typename Compare> static void _sort (size_t sz, T* list, Compare comp)
{
    std::sort(list, list+sz, comp);
}

//------------------------------------------------------------------------------
// Count the number of elements in each bucket
// sz: length of array
// list: array pointer
// np: number of processors (threads)
// range: range of buckets
// bucket: number of items in each bucket (return value)
template <typename T, typename Compare> static void _count (size_t sz, const T* list,
    unsigned np, const T* range, size_t* bucket, Compare comp)
{
    for ( unsigned n = 0; n < np; n++ ) bucket[n] = 0;
    for ( size_t i = 0; i < sz; i++ ) {
        unsigned n;
        for ( n = 0; n < np-1; n++ ) {
            if ( comp(list[i], range[n]) ) {
                bucket[n]++;
                break;
            }
        }
        if (n == np-1) bucket[np-1]++;
    }
}

//------------------------------------------------------------------------------
// Reorder array to separate buckets
// sz: length of array
// list: array pointer
// np: number of processors (threads)
// range: range of buckets
// map: reordering map
// out: reordered array (return value)
// comp: comparator
template <typename T, typename Compare>
static void _reorder (size_t sz, const T* list, unsigned np, const T* range,
					  size_t* map, T* out, Compare comp)
{
    for ( size_t i = 0; i < sz; i++ ) {
        unsigned n;
        for ( n = 0; n < np-1; n++ ) {
            if ( comp(list[i], range[n]) ) {
                out[map[n]++] = list[i];
                break;
            }
        }
        if ( n == np-1 ) out[map[np-1]++] = list[i];
    }
}

//------------------------------------------------------------------------------
// Parallel sort
// sz: length of array
// list: array pointer
// comp: comparator
// np: number of processors (threads)
// sf: sampling factor (100 should be good enough for most cases)
template <typename T, typename Compare = std::less<T> >
void parasort(size_t sz, T* list, Compare comp = Compare(), unsigned np = 2, unsigned sf = 100)
{
	unsigned nSample = np * sf;
	unsigned nMap = np * np;
	std::vector<T> range(np);
	std::vector<T> sample(nSample);
	std::vector<size_t> tmp(nMap, 0), map(nMap, 0), bucket(nMap, 0);
	std::vector<std::thread> threads(np);
// Step 1: Pick np*sf random elements from the input array.
// We use these elements to split the array into even partitions.
	for (unsigned i = 0; i < nSample; i++) sample[i] = list[rand()%sz];
		_sort(nSample, &sample[0], comp);
	for (unsigned i = 0; i < np-1; i++) range[i] = sample[(i+1)*sf];
// Step 2: Count how many elements go into each bucket.
// This is done in parallel
	size_t part = sz / np;
	for (unsigned i = 0; i < np; i++) {
		size_t start = i * part;
		size_t length = ( i+1 == np ) ? sz - i * part : part;
		threads[i] = std::thread( _count<T, Compare>, length, &list[start], np, &range[0], &bucket[i*np],
													  comp );
	}
	for (auto&& thread : threads) thread.join();
// Step 3: Create a mapping scheme to separate partitions.
// Since the number of operations is small, we do this step in serial.
	for (unsigned i = 0; i < nMap; i++) tmp[i] = i ? tmp[i-1] + bucket[((i-1)%np)*np+(i-1)/np] : 0;
	for (unsigned i = 0; i < nMap; i++) map[i] = tmp[(i%np)*np+i/np];
	for (unsigned i = 0; i < nMap; i++) tmp[i] = map[i];
// Step 4: Map the input array into np separate partitions.    
	std::vector<T> sorted(sz);
	for (unsigned i = 0; i < np; i++) {
		size_t start = i * part;
		size_t length = (i+1 == np) ? sz - i * part : part;
		threads[i] = std::thread( _reorder<T, Compare>, length, &list[start], np, &range[0], &tmp[i*np],
														&sorted[0], comp );
	}
	for (auto&& thread : threads) thread.join();
// Step 5: Sort each partition in a separate thread.    
	for (unsigned i = 0; i < np; i++) {
		size_t start = map[i];
		size_t length = (i+1 == np) ? sz - map[i] : map[i+1] - map[i];
		threads[i] = std::thread( _sort<T, Compare>, length, &sorted[start], comp );
	}
	for (auto&& thread : threads) thread.join();
// Step 6: Copy the temporary container for the sorted array into the input array.    
	std::memcpy(list, &sorted[0], sz * sizeof(T) );
}

#endif // !PARASORT_H