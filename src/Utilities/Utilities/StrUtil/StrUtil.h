#ifndef STRUTILS_H_
#define STRUTILS_H_

#include <string>
#include <list>
#include <queue>
#include <stdexcept>
#include <sstream>

namespace utils {

    class bad_cast : public std::logic_error {
        public:
        bad_cast (const std::string& message) : std::logic_error (message) {
        }

        virtual ~bad_cast (void) throw () {
        }
    };

    template < class T > void eatTrim (T& str, const T& lex) {
        if (str.empty ()) {
            return;
        }
        unsigned int pos = str.find_first_not_of (lex);
        if (pos == str.npos) {
            str.clear ();
        } else {
            str = str.substr (pos, str.npos);
        }
    }

    template < class T > static std::list< T > split (const T& source, const T& delimiter) {
        std::list< T > result;
        T temp = source;
        size_t pos = 0;
        do {
            pos = temp.find (delimiter);
            if (pos != temp.npos) {
                if (pos != 0) {
                    result.push_back (temp.substr (0, pos));
                }

                if (pos + delimiter.length () <= temp.length ()) {
                    temp = temp.substr (pos + delimiter.length ());
                } else {
                    result.push_back (temp.substr (0, pos));
                    break;
                }
            } else if (!temp.empty ()) {
                result.push_back (temp);
            }
        } while (pos != temp.npos);

        return result;
    }

    template < class T > T trim (const T& str) {
        T result;
        unsigned int pos = str.find_first_not_of (L" \r\n\t");
        if (pos != str.npos) {
            result = str.substr (pos, str.npos);
        }

        pos = result.find_last_not_of (L" \r\n\t") + 1;
        if (pos != result.npos) {
            result = result.erase (pos);
        }

        return result;
    }

    template < class Out, class In > static Out lexical_cast (const In& inputValue) {
        Out result;

        std::stringstream stream (std::stringstream::in | std::stringstream::out);
        stream << inputValue;
        stream >> result;
        if (stream.fail () || !stream.eof ()) {
            throw bad_cast ("Cast failed");
        }

        return result;
    }


    namespace FileSystem {

        static const std::string CONST_PATH_DELIMITERS = std::string ("/\\");

        template < class T > T getFileName (const T& path) {
            T result;

            size_t pos = path.find_last_of (T (CONST_PATH_DELIMITERS.begin (), CONST_PATH_DELIMITERS.end ()));
            if (pos != path.npos) {
                result = path.substr (pos + 1, path.npos);
            }

            return result;
        }

        template < class T > T getDirectory (const T& path) {
            T result;

            size_t pos = path.find_last_of (T (CONST_PATH_DELIMITERS.begin (), CONST_PATH_DELIMITERS.end ()));
            if (pos != path.npos) {
                result = path.substr (0, pos + 1);
            }

            return result;
        }
    }
}

#endif // STRUTILS_H_
