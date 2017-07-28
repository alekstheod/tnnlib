#ifndef FACTORY_H
#define FACTORY_H

#include <Utilities/MPL/Tuple.h>
#include <map>
#include <functional>


namespace utils {

    template < typename KeyType, typename ObjectType > class Factory {
        public:
        void registerAllocator (KeyType key, std::function< ObjectType*() > allocator) {
            m_allocators[key] = allocator;
        }

        ObjectType* create (KeyType key) {
            ObjectType* result = nullptr;
            if (m_allocators.find (key) != m_allocators.end ()) {
                result = m_allocators[key]();
            }

            return result;
        }

        private:
        std::map< KeyType, std::function< ObjectType*() > > m_allocators;
    };

    template < typename I > struct NewAllocator {
        typedef I type;
        template < typename... Args > I* operator() (Args... args) {
            return new I (std::forward< Args > (args)...);
        }
    };


    template < typename BaseType, template < typename > class Allocator = NewAllocator, typename... T > class StaticFactory {
        public:
        template < typename Type, typename... Args > BaseType* create (Args... args) {
            return utils::get< Allocator< Type > > (m_creators) (std::forward< Args > (args...)...);
        }

        private:
        std::tuple< Allocator< T >... > m_creators;
    };
}

#endif // FACTORY_H
