#ifndef LRUCACHE_H
#define LRUCACHE_H
#include <unordered_map>
#include <list>
#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/bind.hpp>

namespace utils {

    /**
     * Implementation of the Least Recently used cache algorithm.
     */
    template< typename KeyType, typename ValueType >
    class LRUCache {
      private:
        typedef typename std::list< KeyType >::iterator LruIterator;

        /**
         * List of the least recently used keys.
         */
        std::list< KeyType > m_lruList;

        /**
         * A mapping of the keys and the cached data.
         */
        typedef std::unordered_map< KeyType, std::pair< ValueType, LruIterator > > Cache;
        Cache m_cache;

        /**
         * A maximum size of the cache.
         */
        const unsigned int m_capacity;

        /**
         * Default destoryer function.
         */
        void drop(ValueType value) {
        }

        std::pair< KeyType, ValueType& > lruTransform(KeyType key) {
            return std::make_pair(key, std::ref(m_cache[key].first));
        }

        using Iterator = decltype(boost::make_transform_iterator(
         m_lruList.begin(), boost::bind(&LRUCache::lruTransform, (LRUCache*)nullptr, _1)));

      public:
        explicit LRUCache(const unsigned int size)
         : m_capacity(size != 0 ? size : 1) {
        }

        Iterator begin() {
            return boost::make_transform_iterator(
             m_lruList.begin(), boost::bind(&LRUCache::lruTransform, this, _1));
        }

        Iterator end() {
            return boost::make_transform_iterator(
             m_lruList.end(), boost::bind(&LRUCache::lruTransform, this, _1));
        }

        Iterator rbegin() {
            return boost::make_transform_iterator(m_lruList.rbegin(),
                                                  boost::bind(&LRUCache::lruTransform, this, _1));
        }

        Iterator rend() {
            return boost::make_transform_iterator(
             m_lruList.rend(), boost::bind(&LRUCache::lruTransform, this, _1));
        }

        /**
         * Algorithm implementation.
         * @param key key needed for the algorithm.
         * @param creator value creator.
         * @return cached or the new created value.
         */
        template< typename Creator >
        ValueType read(KeyType key, Creator creator) {
            return read(key, creator, std::bind(&LRUCache::drop, this, std::placeholders::_1));
        }

        /**
         * Algorithm implementation.
         * @param key key needed for the algorithm.
         * @param creator value creator.
         * @param destroyer the deallocator for the object.
         * @return cached or the new created value.
         */
        template< typename Creator, typename DropCallback >
        ValueType read(KeyType key, Creator creator, DropCallback drop) {
            ValueType result;
            auto i = m_cache.find(key);
            if(i == m_cache.end()) {
                if(m_cache.size() >= m_capacity) {
                    i = m_cache.find(m_lruList.back());
                    drop(i->first, i->second.first);
                    m_cache.erase(i);
                    m_lruList.pop_back();
                }

                ValueType value = creator(key);
                m_lruList.push_front(key);
                m_cache.insert(
                 std::make_pair(key, std::make_pair(value, m_lruList.begin())));
                result = value;
            } else {
                auto value = i->second;
                if(value.second != m_lruList.begin()) {
                    m_lruList.splice(m_lruList.begin(), m_lruList, value.second);
                }

                result = value.first;
            }

            return result;
        }

        void clear() {
            clear(drop);
        }

        template< typename DropCallback >
        void clear(DropCallback drop) {
            for(auto i = m_cache.begin(); i != m_cache.end(); i++) {
                drop(i->first, i->second.first);
            }

            m_cache.clear();
            m_lruList.clear();
        }

        unsigned int size() {
            return m_lruList.size();
        }

        ~LRUCache() {
        }
    };
} // namespace utils

#endif // LRUCACHE_H
