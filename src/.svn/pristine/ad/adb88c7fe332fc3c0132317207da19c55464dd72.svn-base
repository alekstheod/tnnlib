/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with
*  or without modification, are permitted provided that the
*  following conditions are met:
*  Redistributions of source code must retain the above
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#ifndef OBSERVABLE_H
#define OBSERVABLE_H
#include <set>

namespace utils {

/**
 * @author alekstheod
 * The Observer design
 * pattern implementation.
 */
template<class T>
class Observable
{
private:
    bool operator==(const Observer& other) const;
    Observer(const Observer& other);

private:
    /**
     * List of the listeners, objects
     * of observer.
     */
    std::set<T*> _listeners;

protected:
      /**
    * Empty constructor
    * will initialize the object.
    */
    Observable() {
    }
    
    std::set<T*> getListeners() {
        return _listeners;
    }

    /**
     * Destructor.
     */
    ~Observer() {
    }
    
public:
    /**
    * Assign operator overload.
    */
    Observable& operator=(const Observable& other) {
        _listeners = other._listeners;
    }

    /**
     * Will add a new listener
     * to the listeners list. The listener
     * will not be added in case if it is
     * already present in the listeners list.
     * @param listener instance of the listener.
     * @return true if succeed, false otherwise.
     */
    bool addListener(T* listener) {
        bool result = false;

        if( _listeners.find(listener) == _listeners.end() ) {
            _listeners.insert(listener);
            result = true;
        }

        return result;
    }
    
    /**
     * Will remove a listener
     * from the listeners list
     * @param listener instance of the listener.
     * @return true if succeed, false otherwise.
     */
    bool removeListener( T* listener ) {
        bool result = false;
        if( _listeners.find(listener) == _listeners.end() ) {
            _listeners.erase(listener);
            result = true;
        }

        return result;
    }
};

}

#endif // OBSERVER_H
